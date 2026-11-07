# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""EVAL (BUILDER 7, 9): held-out quality on ALL held-out views and the render-based gates, from one trained model.

Held-out (7.1): raw PSNR (identity exposure, sky panorama composited: the trained model's own rendering of the photo),
  raw PSNR without the sky panorama (black background: what today's viewer shows), PSNR on non-sky pixels only,
  exposure-interpolated PSNR (labelled), SSIM (our 11x11 Gaussian code), each with and without the pre-pass exclusions,
  plus per-stretch values: 50 m of path per stretch, merged with a neighbour until each holds >= 3 held-out views.
Every keyframe (V2.1 coverage, V2.3 PSNR dips), off-trajectory renders at every 5th keyframe (V3.6: +-0.5 / +-1 m
lateral, +2 m up, the viewer's FOLLOW pose 6.8 m back / 2.2 m up; near-camera occupancy), V3.1 floaters, V3.7 transients.
Renders use the trainer's rasteriser (gsplat on the GPU; the reference rasteriser in CPU tests).
"""
from __future__ import annotations

import math
import os

import json
import os
import numpy as np
import torch

from . import exposure as EX
from . import io_lib as IO
from . import losses as LS
from . import masks as MK
from . import sky as SKY
from . import util as U
from .render import look_at

GATE_BARS = dict(v21_alpha=0.5, v21_good=0.95, v21_good_share=0.99, v21_run_bad=0.90, v21_run_len=5,
                 v23_drop_db=3.0, v23_run_len=5, v23_win=50, v31_ratio=0.8, v31_view_median_max=0.02, v31_view_max=0.10,
                 v31_min_disp_px=1.0,
                 v23_static_pixels=True,    # 2c amendment 2026-10-22 (lead; OAK 164424): V2.3 on static pixels (V2.1's set); full image reported
                 # (v31_min_disp_px above) 2c amendment 2026-10-22: a floater must also be >= 1 px of disparity in front
                 pa_floor_db=20.0, pa_stretch_drop_db=4.0,   # 2c amendment: P-a replaced (V3_VS_V4.md)
                 sky6_edge_px=7,   # 2c: SKY-6 decides on sky farther than 7 px from non-sky (the SKY-NEW-1 band width)
                 align_max_deg=5.0, align_steps_deg=(1.0, 0.5, 0.25, 0.1),   # 2c: held-out rotation alignment (map frozen)
                 v36_drop_max=0.10, near_m=2.0, stretch_m=50.0, stretch_min_views=3, follow_back_m=6.8, follow_up_m=2.2,
                 v37_tol_px=1.0, v37_tol_rel=0.10,
                 # SKY-6 (phase 2a): what the site shows in sky directions (splats on a black background)
                 sky_alpha_mean_max=0.03, sky_alpha_p95_max=0.10, curtain_deg=10.0, curtain_alpha_mean_max=0.05)


class Model:
    def __init__(self, params, renderer, device, sky=None, expo=None, t_train=None, sh_degree=1, label="trained"):
        self.params, self.render, self.device = params, renderer, torch.device(device)
        self.sky, self.expo, self.t_train, self.sh, self.label = sky, expo, t_train, sh_degree, label

    @torch.no_grad()
    def view(self, T_wc, K, W, H, with_sky=True):
        Tcw = torch.as_tensor(IO.inv_T(T_wc), dtype=torch.float32, device=self.device)
        Kt = torch.as_tensor(K, dtype=torch.float32, device=self.device)
        col, al, _ = self.render(self.params, Tcw, Kt, int(W), int(H), self.sh, depth=True)
        rgb = col[..., :3]
        if with_sky and self.sky is not None:
            rgb = SKY.composite(rgb, al, self.sky.sample(SKY.ray_dirs_world(T_wc, K, W, H, self.device)))
        return rgb.clamp(0, 1).cpu().numpy(), al[..., 0].cpu().numpy(), col[..., 3].cpu().numpy()


def params_to_torch(np_params, device):
    return {k: torch.as_tensor(np.ascontiguousarray(v), device=device) for k, v in np_params.items()}


def _img(prep, j):
    a = np.asarray(prep.img[j], np.float32) / 255.0
    return np.repeat(a, 3, axis=2) if a.shape[2] == 1 else a


def path_s(prep):
    P = np.array([prep.T_wc(v["j"])[:3, 3] for v in prep.views])
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def stretches(s_views, every_m=50.0, min_views=3):
    """Distance-based stretches (7.1): bins of every_m of path, merged with a neighbour until each holds >= min_views."""
    if not len(s_views):
        return []
    edges = list(np.arange(0.0, float(max(s_views)) + every_m, every_m))
    bins = [[a, b] for a, b in zip(edges[:-1], edges[1:])]
    def count(b):
        return int(((s_views >= b[0]) & (s_views < b[1])).sum())
    changed = True
    while changed and len(bins) > 1:
        changed = False
        for i, b in enumerate(bins):
            if count(b) < min_views:
                j = i - 1 if i > 0 else i + 1
                lo, hi = min(bins[i][0], bins[j][0]), max(bins[i][1], bins[j][1])
                bins[min(i, j)] = [lo, hi]
                del bins[max(i, j)]
                changed = True
                break
    return bins


def floater_mask(ed, disp, fB, ratio=None, min_px=None):
    """V3.1 per pixel: (floater, front). front = rendered depth < ratio x RAFT depth (fB / disp); floater = front AND the
    rendered surface is >= min_px of disparity in front of RAFT (2c amendment 2026-10-22: at 20-40 m on Rosario the 0.8 x
    depth test is a 0.26-0.52 px disparity difference, below RAFT's precision)."""
    ratio = GATE_BARS["v31_ratio"] if ratio is None else ratio
    min_px = GATE_BARS["v31_min_disp_px"] if min_px is None else min_px
    z = fB / np.maximum(disp, 1e-6)
    front = ed < ratio * z
    return front & (fB / np.maximum(ed, 1e-3) - disp >= min_px), front


def pa_gate(all_, stretches_):
    """P-a as amended 2026-10-22 (THRESHOLDS 10, V3_VS_V4.md): all-held-out raw PSNR (panorama composited) >= 20.0 dB AND no
    50-m stretch more than 4 dB below the route median of the stretches."""
    if not all_:
        return dict(pass_=False, why="no held-out views")
    med = float(np.median([x["psnr_raw"] for x in stretches_])) if stretches_ else None
    worst = min((x["psnr_raw"] for x in stretches_), default=None)
    ok_floor = all_["psnr_raw"] >= GATE_BARS["pa_floor_db"]
    ok_st = med is None or worst >= med - GATE_BARS["pa_stretch_drop_db"]
    return dict(psnr_raw=all_["psnr_raw"], floor_db=GATE_BARS["pa_floor_db"], route_median_stretch=med, stretch_min=worst,
                stretch_drop_max_db=GATE_BARS["pa_stretch_drop_db"], pass_=bool(ok_floor and ok_st))


def _rot(axis, deg):
    a = np.radians(deg); c, s_ = np.cos(a), np.sin(a)
    R = np.eye(4)
    i, k = [(1, 2), (2, 0), (0, 1)][axis]
    R[i, i], R[i, k], R[k, i], R[k, k] = c, -s_, s_, c
    return R


def align_rotation(model, prep, j, img, mask=None):
    """2c (2026-10-22, work/_diag_eval/pitch_scan.json): the held-out view's camera ROTATION re-estimated against the FROZEN
    map (pitch, yaw, roll about the camera axes; coordinate search at 1 / 0.5 / 0.25 / 0.1 deg, each angle <= 5 deg), by the
    PSNR of the full rendering (panorama composited) on the photo. The position is the SLAM one. Returns (T_wc, angles, psnr0,
    psnr). Map-fault gates (V3.1, SKY-6) are measured at this pose: a 1-3 deg keyframe pitch error of the SLAM pose put the
    tree line on the photo's sky and made every gate fail on kf 260 / 610 / 810 although the map is right there."""
    K, W, H = prep.K, prep.W, prep.H
    T0 = prep.T_wc(j)
    def score(ang):
        T = T0 @ _rot(0, ang[0]) @ _rot(1, ang[1]) @ _rot(2, ang[2])
        rgb, _, _ = model.view(T, K, W, H, with_sky=True)
        return IO.psnr_np(rgb, img, mask) if mask is not None else IO.psnr_np(rgb, img)
    ang = [0.0, 0.0, 0.0]
    best = p0 = score(ang)
    lim = GATE_BARS["align_max_deg"]
    for st in GATE_BARS["align_steps_deg"]:
        improved = True
        while improved:
            improved = False
            for ax in (0, 1, 2):
                for sg in (1, -1):
                    cand = list(ang); cand[ax] = round(cand[ax] + sg * st, 4)
                    if abs(cand[ax]) > lim + 1e-9:
                        continue
                    sc = score(cand)
                    if sc > best + 1e-4:
                        best, ang, improved = sc, cand, True
    T = T0 @ _rot(0, ang[0]) @ _rot(1, ang[1]) @ _rot(2, ang[2])
    return T, ang, p0, best


def sky6_split(sky6_all, edge_px=None):
    """(interior, edge) of the SKY-6 set: the edge is the sky within edge_px of a non-sky pixel. 2c amendment
    (2026-10-22, work/_arms/sky_resid_class.json + the cap20_sky6 arm): after the sky fixes 81 % of the residual sky alpha
    sits in that band at the tree line, and training it as sky cost 1.4 dB of probe PSNR in 4000 steps (it holds tree-top
    content the panorama cannot draw). The edge alpha is reported, the decision is on the interior."""
    import cv2
    edge_px = GATE_BARS["sky6_edge_px"] if edge_px is None else edge_px
    if not sky6_all.any():
        return sky6_all, sky6_all
    dist = cv2.distanceTransform(sky6_all.astype(np.uint8), cv2.DIST_L2, 3)
    return sky6_all & (dist > edge_px), sky6_all & (dist <= edge_px)


def _gate_metrics(model, prep, Twc, b, sky, sky6m, el, d):
    """V3.1 and SKY-6 quantities of one view at camera pose Twc (masks belong to the photo)."""
    K, W, H = prep.K, prep.W, prep.H
    _, al, ed = model.view(Twc, K, W, H, with_sky=False)
    out = {}
    if sky6m.any():
        band = sky6m & (el < np.sin(np.radians(GATE_BARS["curtain_deg"])))
        out.update(sky_alpha_mean=float(al[sky6m].mean()), sky_alpha_p95=float(np.percentile(al[sky6m], 95)),
                   curtain_alpha_mean=float(al[band].mean()) if band.any() else None)
    ok = ((b & LS.BIT_VALID) != 0) & ((b & (LS.BIT_SKY | LS.BIT_SKY_MAYBE | LS.BIT_SAT | LS.BIT_EDGE)) == 0) & (al > 0.5)
    if ok.any():
        fl_m, front = floater_mask(ed[ok], d[ok], prep.fB)
        out["floater_share"] = float(fl_m.mean())
        out["floater_share_unrestricted"] = float(front.mean())
    return out


def _stretch_table(rows):
    s_v = np.array([r["s"] for r in rows])
    st = []
    if not len(s_v):
        return st
    for lo, hi in stretches(s_v, GATE_BARS["stretch_m"], GATE_BARS["stretch_min_views"]):
        sel = [r for r in rows if lo <= r["s"] < hi or (hi == max(s_v, default=0) and r["s"] == hi)]
        if sel:
            st.append(dict(s0=round(lo, 1), s1=round(hi, 1), n=len(sel), psnr_raw=float(np.mean([r["psnr_raw"] for r in sel])),
                           psnr_raw_nosky=float(np.mean([r["psnr_raw_nosky"] for r in sel])),
                           ssim=float(np.mean([r["ssim"] for r in sel]))))
    return st


def _sky6_v31(rows):
    fl = [r["floater_share"] for r in rows if "floater_share" in r]
    fu = [r["floater_share_unrestricted"] for r in rows if "floater_share_unrestricted" in r]
    sa = [r for r in rows if r.get("sky_alpha_mean") is not None]
    cu = [r["curtain_alpha_mean"] for r in sa if r.get("curtain_alpha_mean") is not None]
    sky6 = dict(n_views=len(sa), sky_alpha_mean=float(np.mean([r["sky_alpha_mean"] for r in sa])) if sa else None,
                sky_alpha_p95_max=float(np.max([r["sky_alpha_p95"] for r in sa])) if sa else None,
                views_p95_over=int(sum(r["sky_alpha_p95"] > GATE_BARS["sky_alpha_p95_max"] for r in sa)),
                views_over_kf=[r.get("kf_id") for r in sa if r["sky_alpha_p95"] > GATE_BARS["sky_alpha_p95_max"]],
                curtain_alpha_mean=float(np.mean(cu)) if cu else None,
                rule="confident-sky pixels + maybe-sky the scan proxy calls sky (phase 2b) of every held-out view, splats alone (black background, as the site shows them): "
                     "mean alpha <= %.2f, every view's p95 <= %.2f; sky within %.0f deg above the horizon (the curtain band): "
                     "mean alpha <= %.2f" % (GATE_BARS["sky_alpha_mean_max"], GATE_BARS["sky_alpha_p95_max"],
                                              GATE_BARS["curtain_deg"], GATE_BARS["curtain_alpha_mean_max"]))
    sky6["pass_"] = bool((not sa) or (sky6["sky_alpha_mean"] <= GATE_BARS["sky_alpha_mean_max"]
                                      and sky6["sky_alpha_p95_max"] <= GATE_BARS["sky_alpha_p95_max"]
                                      and (sky6["curtain_alpha_mean"] is None or sky6["curtain_alpha_mean"] <= GATE_BARS["curtain_alpha_mean_max"])))
    v31 = dict(view_median=float(np.median(fl)) if fl else None, view_max=float(np.max(fl)) if fl else None,
               pass_=bool(fl and np.median(fl) <= GATE_BARS["v31_view_median_max"] and np.max(fl) <= GATE_BARS["v31_view_max"]),
               rule="share of valid non-sky pixels (alpha > 0.5) whose rendered depth < %.1f x RAFT AND >= %.1f px of "
                    "disparity in front (2c amendment 2026-10-22)" % (GATE_BARS["v31_ratio"], GATE_BARS["v31_min_disp_px"]),
               unrestricted=dict(view_median=float(np.median(fu)) if fu else None, view_max=float(np.max(fu)) if fu else None))
    return sky6, v31


def eval_heldout(model, prep, views=None, align=True):
    K, W, H = prep.K, prep.W, prep.H
    win = IO.gauss_win()
    s_all = path_s(prep)
    views = prep.hold_ids if views is None else views
    rows = []
    for j in views:
        v = prep.views[j]
        Twc = prep.T_wc(j)
        img = _img(prep, j)
        b = np.asarray(prep.mask[j])
        rgb, al, ed = model.view(Twc, K, W, H, with_sky=True)
        rgb0, _, _ = model.view(Twc, K, W, H, with_sky=False) if model.sky is not None else (rgb, None, None)
        sky = (b & LS.BIT_SKY) != 0
        anysky = (b & (LS.BIT_SKY | LS.BIT_SKY_MAYBE)) != 0
        rec = dict(j=int(j), kf_id=v["kf_id"], frame_index=v.get("frame_index"), t=v["t"], s=float(s_all[j]),
                   excluded=bool((v.get("exclude") or {}).get("exclude_train")), same_photo=bool(v.get("same_photo")),
                   psnr_raw=IO.psnr_np(rgb, img), psnr_raw_nosky=IO.psnr_np(rgb0, img),
                   psnr_nonsky_px=IO.psnr_np(rgb0, img, ~anysky) if (~anysky).any() else None, sky_share=float(sky.mean()))
        # SKY-6 (phase 2b, SKY-NEW-2): confident sky PLUS maybe-sky that the scan proxy also calls sky (the band above the
        # Rosario tree line, an overcast sky that is never confident); grey rows: the proxy is bright + low texture
        from .masks import proxy_sky, ray_elevation
        el = ray_elevation(Twc, K, W, H, prep.up)
        u8 = np.asarray(prep.img[j])
        u8 = u8[..., 0] if (u8.ndim == 3 and u8.shape[2] == 1) else u8
        prox = proxy_sky(u8, el > -np.sin(np.radians(3.0)))
        sky6_all = sky | (((b & LS.BIT_SKY_MAYBE) != 0) & prox)
        sky6m, sky6_edge = sky6_split(sky6_all)
        if sky6m.any():
            band = sky6m & (el < np.sin(np.radians(GATE_BARS["curtain_deg"])))
            rec.update(sky_alpha_mean=float(al[sky6m].mean()), sky_alpha_p95=float(np.percentile(al[sky6m], 95)),
                       curtain_alpha_mean=float(al[band].mean()) if band.any() else None, curtain_px=int(band.sum()),
                       sky6_px=int(sky6m.sum()), sky6_maybe_px=int((sky6m & ~sky).sum()),
                       sky6_edge_px=int(sky6_edge.sum()), sky6_edge_alpha_mean=float(al[sky6_edge].mean()) if sky6_edge.any() else None)
        with torch.no_grad():
            rec["ssim"] = float(IO.ssim(torch.tensor(rgb).permute(2, 0, 1)[None], torch.tensor(img).permute(2, 0, 1)[None], win))
        if model.expo is not None and model.t_train is not None:
            a_, b_ = model.expo.interpolated(v["t"], model.t_train)
            rec["psnr_expo"] = IO.psnr_np(EX.apply_numpy(rgb, a_, b_), img)
        d = np.asarray(prep.disp[j], np.float32)
        ok = ((b & LS.BIT_VALID) != 0) & ((b & (LS.BIT_SKY | LS.BIT_SKY_MAYBE | LS.BIT_SAT | LS.BIT_EDGE)) == 0) & (al > 0.5)
        if ok.any():
            z = prep.fB / np.maximum(d[ok], 1e-6)
            fl_m, front = floater_mask(ed[ok], d[ok], prep.fB)
            rec["floater_share"] = float(fl_m.mean())
            rec["floater_share_unrestricted"] = float(front.mean())
        if align:
            Ta, ang, p0, p1 = align_rotation(model, prep, j, img)
            rgb_a, _, _ = model.view(Ta, K, W, H, with_sky=True)
            with torch.no_grad():
                ssim_a = float(IO.ssim(torch.tensor(rgb_a).permute(2, 0, 1)[None], torch.tensor(img).permute(2, 0, 1)[None], win))
            rec["aligned"] = dict(angles_deg=ang, psnr_raw=p1, ssim=ssim_a, gain_db=p1 - p0, at_bound=bool(max(abs(x) for x in ang) >= GATE_BARS["align_max_deg"] - 1e-9),
                                  **_gate_metrics(model, prep, Ta, b, sky, sky6m, el, d))
        rows.append(rec)
    def agg(sel):
        if not sel:
            return None
        out = {k: float(np.mean([r[k] for r in sel if r.get(k) is not None])) for k in
               ("psnr_raw", "psnr_raw_nosky", "psnr_nonsky_px", "ssim", "psnr_expo") if any(r.get(k) is not None for r in sel)}
        out["n"] = len(sel)
        return out
    st = _stretch_table(rows)
    rows_all = rows
    rows = [r for r in rows_all if not r["excluded"]]      # gates decided on non-excluded views (lead 2026-10-22, FQ-4)
    sky6, v31 = _sky6_v31([dict(r.get("aligned", r), kf_id=r["kf_id"]) for r in rows] if align else rows)
    sky6_slam, v31_slam = _sky6_v31(rows)
    sky6["all_frames"], v31["all_frames"] = _sky6_v31([dict(r.get("aligned", r), kf_id=r["kf_id"]) for r in rows_all] if align else rows_all)
    if align:
        al_ = [r["aligned"] for r in rows]
        sky6["measured_at"] = v31["measured_at"] = ("held-out camera rotation re-aligned to the frozen map (2c, THRESHOLDS 10.4); "
                                                    "the values at the SLAM pose are under at_slam_pose")
        sky6["at_slam_pose"], v31["at_slam_pose"] = sky6_slam, v31_slam
        angs = [max(map(abs, a["angles_deg"])) for a in al_]
        alignment = dict(angle_abs_deg_median=float(np.median(angs)), angle_abs_deg_max=float(np.max(angs)),
                         pose_issue_flag=bool(np.median(angs) > 2.0),
                         ssim_aligned_mean=float(np.mean([a["ssim"] for a in al_])),
                         method=("gate-measurement aid only (lead 2026-10-22): each held-out camera's rotation (pitch / yaw / roll "
                                 "about its own axes) re-estimated against the FROZEN map by coordinate search (1 / 0.5 / 0.25 / "
                                 "0.1 deg, each <= 5 deg) maximising the full-render PSNR; SLAM position kept; the headline "
                                 "numbers are at the SLAM pose; a median > 2 deg is flagged as a pose issue, not a map issue"),
                         angle_abs_max_deg_p50_p90_max=[round(float(np.percentile([max(map(abs, a["angles_deg"])) for a in al_], q)), 3) for q in (50, 90, 100)],
                         gain_db_mean=float(np.mean([a["gain_db"] for a in al_])), psnr_raw_aligned_mean=float(np.mean([a["psnr_raw"] for a in al_])),
                         n_at_bound=int(sum(a["at_bound"] for a in al_)),
                         note="P-a stays at the SLAM pose (psnr_raw); this is the held-out pose quality, disclosed")
    else:
        alignment = None
    st_dec = _stretch_table(rows)
    p_a = pa_gate(agg(rows), st_dec)
    p_a["decided_on"] = "non-excluded held-out views (%d of %d; lead 2026-10-22, FQ-4)" % (len(rows), len(rows_all))
    p_a["all_frames"] = pa_gate(agg(rows_all), st)
    out = dict(all=agg(rows_all), without_excluded=agg(rows),
               same_photo=agg([r for r in rows_all if r["same_photo"]]), per_stretch=st, per_stretch_without_excluded=st_dec,
               stretch_min=min((x["psnr_raw"] for x in st), default=None),
               route_median_stretch=float(np.median([x["psnr_raw"] for x in st])) if st else None, views=rows_all,
               sky6=sky6, v31=v31, alignment=alignment,
               p_a=p_a,
               headline=("raw PSNR = identity exposure, sky panorama composited; psnr_raw_nosky = the splat map alone on a "
                         "black background (today's viewer); psnr_expo = exposure-interpolated (labelled)"))
    return out


def _v21_v23(out, key=None):
    cov = np.array([o["coverage"] for o in out])
    good = np.nan_to_num(cov, nan=1.0) >= GATE_BARS["v21_good"]
    bad = np.nan_to_num(cov, nan=1.0) < GATE_BARS["v21_run_bad"]
    runs, i = [], 0
    while i < len(bad):
        if bad[i]:
            k = i
            while k + 1 < len(bad) and bad[k + 1]:
                k += 1
            if k - i + 1 >= GATE_BARS["v21_run_len"]:
                runs.append(dict(t0=out[i]["t"], t1=out[k]["t"], n=k - i + 1, min=float(np.nanmin(cov[i:k + 1]))))
            i = k + 1
        else:
            i += 1
    key = key or ("psnr_static" if GATE_BARS.get("v23_static_pixels") else "psnr")
    ps = np.array([o[key] if o.get(key) is not None else o["psnr"] for o in out])
    w = GATE_BARS["v23_win"]
    med = np.array([np.median(ps[max(0, i - w):i + w + 1]) for i in range(len(ps))])
    dip = ps < med - GATE_BARS["v23_drop_db"]
    dips, i = [], 0
    while i < len(dip):
        if dip[i]:
            k = i
            while k + 1 < len(dip) and dip[k + 1]:
                k += 1
            if k - i + 1 >= GATE_BARS["v23_run_len"]:
                thr = med[i:k + 1] - GATE_BARS["v23_drop_db"]
                dips.append(dict(t0=out[i]["t"], t1=out[k]["t"], n=k - i + 1, kf0=out[i].get("kf_id"), kf1=out[k].get("kf_id"),
                                 min_psnr_db=float(ps[i:k + 1].min()), max_db_under_threshold=float((thr - ps[i:k + 1]).max())))
            i = k + 1
        else:
            i += 1
    return dict(v21=dict(good_share=float(good.mean()), runs_below_090=runs,
                                     pass_=bool(good.mean() >= GATE_BARS["v21_good_share"] and not runs)),
                v23=dict(dips=dips, pass_=not dips))


def _is_excluded(v):
    return bool((v.get("exclude") or {}).get("exclude_train")) or v.get("role") in ("excl", "excluded")


def eval_every_kf(model, prep):
    K, W, H = prep.K, prep.W, prep.H
    out = []
    for v in prep.views:
        j = v["j"]
        rgb, al, _ = model.view(prep.T_wc(j), K, W, H, with_sky=True)
        b = np.asarray(prep.mask[j])
        m = ((b & LS.BIT_VALID) != 0) & ((b & (LS.BIT_SKY | LS.BIT_SKY_MAYBE | LS.BIT_SAT | LS.BIT_MOVING | LS.BIT_CAMFIXED)) == 0)
        cov = float((al[m] >= GATE_BARS["v21_alpha"]).mean()) if m.any() else float("nan")
        stat = (b & (LS.BIT_MOVING | LS.BIT_CAMFIXED)) == 0
        ph = _img(prep, j)
        out.append(dict(j=j, kf_id=v["kf_id"], t=v["t"], role=v["role"], excluded=_is_excluded(v), coverage=cov, psnr=IO.psnr_np(rgb, ph),
                        psnr_static=IO.psnr_np(rgb, ph, stat) if stat.any() else None,
                        alpha_full=float((al >= GATE_BARS["v21_alpha"]).mean())))
    dec = [o for o in out if not o["excluded"]]
    g = _v21_v23(dec)
    g_all = _v21_v23(out)
    g["v21"]["all_frames"], g["v23"]["all_frames"] = g_all["v21"], g_all["v23"]
    g["v23"]["full_image"] = _v21_v23(dec, "psnr")["v23"]
    g["v23"]["measured_on"] = ("static pixels (moving / rig-fixed masked, V2.1's set; 2c amendment 2026-10-22); the full-image "
                               "result is under full_image") if GATE_BARS.get("v23_static_pixels") else "full image"
    g["v21"]["decided_on"] = g["v23"]["decided_on"] = ("non-excluded keyframes (%d of %d; FQ-excluded frames are reported under "
                                                       "all_frames: lead 2026-10-22)" % (len(dec), len(out)))
    return dict(per_kf=out, **g)


def follow_pose(T_wc, up):
    c = T_wc[:3, 3]
    f = T_wc[:3, 2] - (T_wc[:3, 2] @ up) * up
    f = f / max(np.linalg.norm(f), 1e-9)
    eye = c - GATE_BARS["follow_back_m"] * f + GATE_BARS["follow_up_m"] * up
    return IO.inv_T(look_at(eye, c + 2.0 * f, up))


def eval_off_trajectory(model, prep, every=5):
    K, W, H = prep.K, prep.W, prep.H
    up = prep.up / np.linalg.norm(prep.up)
    rows = []
    for v in prep.views[::every]:
        T = prep.T_wc(v["j"])
        _, al0, _ = model.view(T, K, W, H, with_sky=False)
        c0 = float((al0 >= 0.5).mean())
        rec = dict(kf_id=v["kf_id"], t=v["t"], on=c0)
        for name, off in (("lat-1", -1.0), ("lat-0.5", -0.5), ("lat+0.5", 0.5), ("lat+1", 1.0)):
            T2 = T.copy()
            T2[:3, 3] = T[:3, 3] + off * T[:3, 0]
            rec[name] = float((model.view(T2, K, W, H, False)[1] >= 0.5).mean())
        T3 = T.copy()
        T3[:3, 3] = T[:3, 3] + 2.0 * up
        rec["up+2"] = float((model.view(T3, K, W, H, False)[1] >= 0.5).mean())
        Tf = follow_pose(T, up)
        _, alf, edf = model.view(Tf, K, W, H, False)
        rec["follow"] = float((alf >= 0.5).mean())
        rec["follow_near_occupancy"] = float(((edf < GATE_BARS["near_m"]) & (alf > 0.5)).mean())
        rows.append(rec)
    drops = {k: float(np.mean([r["on"] - r[k] for r in rows])) for k in ("lat-1", "lat-0.5", "lat+0.5", "lat+1", "up+2", "follow")} if rows else {}
    return dict(per_kf=rows, mean_drop=drops, follow_near_occupancy_median=float(np.median([r["follow_near_occupancy"] for r in rows])) if rows else None,
                pass_=bool(rows and drops.get("follow", 1) <= GATE_BARS["v36_drop_max"]),
                note="near-camera occupancy bar is calibrated in 0e on the clean maps (INFERENCE until then)")


def eval_transients(model, prep):
    """V3.7: share of held-out pixels whose rendered depth disagrees with RAFT disparity reprojected from BOTH time
    neighbours (training keyframes) - V3.1 cannot see these (it compares with the same frame's RAFT depth)."""
    K, W, H = prep.K, prep.W, prep.H
    tr = prep.train_ids
    out = []
    for j in prep.hold_ids:
        prev = [k for k in tr if k < j]
        nxt = [k for k in tr if k > j]
        nb = ([prev[-1]] if prev else []) + ([nxt[0]] if nxt else [])
        if len(nb) < 2:
            continue
        _, al, ed = model.view(prep.T_wc(j), K, W, H, False)
        dis = np.ones((H, W), bool)
        seen = np.zeros((H, W), bool)
        for k in nb:
            ub, vb, dpred, ok = MK.reproject_disparity(np.asarray(prep.disp[k], np.float32), prep.T_wc(k), prep.T_wc(j), K, prep.fB)
            ui, vi = np.rint(ub[ok]).astype(int), np.rint(vb[ok]).astype(int)
            e = ed[vi, ui]
            a = al[vi, ui] > 0.5
            dr = prep.fB / np.maximum(e, 1e-6)
            tol = np.maximum(GATE_BARS["v37_tol_px"], GATE_BARS["v37_tol_rel"] * dpred[ok])
            agree = np.zeros((H, W), bool)
            agree[vi[a], ui[a]] = np.abs(dr[a] - dpred[ok][a]) <= tol[a]
            hit = np.zeros((H, W), bool)
            hit[vi[a], ui[a]] = True
            dis &= ~(hit & agree)
            seen |= hit
        out.append(dict(kf_id=prep.views[j]["kf_id"], share=float((dis & seen).sum() / max(seen.sum(), 1))))
    sh = [o["share"] for o in out]
    return dict(per_view=out, median=float(np.median(sh)) if sh else None, max=float(np.max(sh)) if sh else None,
                note="bar calibrated in 0e on the clean vs faulty maps (INFERENCE until then)")


# ---------------------------------------------------------------------------- the staging viewer's sky backdrop (exact port)
VIEWER_CLEAR_RGB = (4 / 255.0, 6 / 255.0, 10 / 255.0)   # staging viewer clear colour 0x04060a
VIEWER_FADE_DEG = 4.0
VIEWER_FEATHER_TX = 6


def viewer_backdrop_texture(pano_u8, mask_u8, min_lat_deg):
    """staging_site/js/pages/viewer/sky.js fillPanorama, ported 1:1: unobserved sky texels (above min_lat) take the mean
    observed colour of their row (nearest observed row), feathered (+-6 texels) into the observed ones; rows below min_lat
    repeat the lowest sky row. Returns uint8 [H, W, 3] or None (no observed sky: the viewer then uses the raw export)."""
    P = pano_u8.astype(np.float64)
    H, W = P.shape[:2]
    M = mask_u8 if mask_u8.ndim == 2 else mask_u8.max(2)
    lat = 90 - (np.arange(H) + 0.5) / H * 180
    obs = (M > 0) & (P.max(2) > 0)
    mean = np.zeros((H, 3)); has = np.zeros(H, bool)
    for r in range(H):
        if lat[r] < min_lat_deg:
            continue
        o = obs[r]
        if o.any():
            has[r] = True
            mean[r] = P[r][o].mean(0)
    rows = np.nonzero(has)[0]
    if not len(rows):
        return None
    for r in range(H):
        if has[r] or lat[r] < min_lat_deg:
            continue
        mean[r] = mean[rows[np.argmin(np.abs(rows - r))]]
    out = np.zeros((H, W, 3), np.uint8)
    low = -1
    k = np.ones(2 * VIEWER_FEATHER_TX + 1)
    for r in range(H):
        if lat[r] < min_lat_deg:
            if low >= 0:
                out[r] = out[low]
            continue
        low = r
        o = obs[r].astype(np.float64)
        s = np.convolve(np.r_[o[-VIEWER_FEATHER_TX:], o, o[:VIEWER_FEATHER_TX]], k, mode="valid")
        w = np.where(obs[r], s / (2 * VIEWER_FEATHER_TX + 1), 0.0)[:, None]
        out[r] = np.floor(w * P[r] + (1 - w) * mean[r][None, :] + 0.5).clip(0, 255).astype(np.uint8)   # Math.round
    return out


class ViewerBackdrop:
    """The backdrop exactly as the staging viewer draws it (sky.js shader): direction -> (e1, e2, up) basis, lat clamped
    to >= min_lat, bilinear (circular in longitude, clamped in latitude, texel centres), then mixed to the clear colour by
    smoothstep(0, FADE, min_lat - lat)."""

    def __init__(self, md, device="cuda"):
        import cv2
        meta = json.load(open(os.path.join(md, "sky_equirect.json")))
        pano = cv2.imread(os.path.join(md, "sky_equirect.png"), cv2.IMREAD_UNCHANGED)
        if pano.ndim == 2:
            pano = np.repeat(pano[..., None], 3, 2)
        else:
            pano = pano[..., ::-1][..., :3]
        mask = cv2.imread(os.path.join(md, "sky_observed.png"), cv2.IMREAD_UNCHANGED)
        self.min_lat = float((meta.get("below_horizon_blanked") or {}).get("min_lat_deg", -90.0))
        tex = viewer_backdrop_texture(np.ascontiguousarray(pano), mask, self.min_lat) if mask is not None else None
        self.filled = tex is not None
        tex = tex if tex is not None else pano
        self.tex = torch.as_tensor(tex.astype(np.float32) / 255.0, device=device)        # [H, W, 3]
        B = meta["basis_rows"]
        self.basis = torch.tensor([B["e1"], B["e2"], B["up"]], dtype=torch.float32, device=device)
        self.clear = torch.tensor(VIEWER_CLEAR_RGB, dtype=torch.float32, device=device)
        self.device = device

    def sample(self, dirs):
        H, W = self.tex.shape[:2]
        d = torch.nn.functional.normalize(dirs, dim=-1) @ self.basis.T
        lon = torch.atan2(d[..., 1], d[..., 0])
        lat = torch.asin(d[..., 2].clamp(-1, 1))
        latS = torch.clamp(lat, min=np.radians(self.min_lat))
        u = lon / np.pi * 0.5 + 0.5
        v = 0.5 - latS / np.pi
        x = u * W - 0.5
        y = (v * H - 0.5).clamp(0, H - 1)
        x0 = torch.floor(x); y0 = torch.floor(y).clamp(0, H - 1)
        fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
        x0i = x0.long() % W; x1i = (x0i + 1) % W
        y0i = y0.long(); y1i = (y0i + 1).clamp(max=H - 1)
        c = (self.tex[y0i, x0i] * (1 - fx) * (1 - fy) + self.tex[y0i, x1i] * fx * (1 - fy)
             + self.tex[y1i, x0i] * (1 - fx) * fy + self.tex[y1i, x1i] * fx * fy)
        t = ((np.radians(self.min_lat) - lat) / np.radians(VIEWER_FADE_DEG)).clamp(0, 1)
        k = (t * t * (3 - 2 * t))[..., None]                                             # GLSL smoothstep
        return c * (1 - k) + self.clear * k


GRADIENT_BACKDROP = dict(type="gradient", horizon_grey=0.80, zenith_grey=0.62, ramp="linear in sin(elevation)",
                         below_horizon_fade_deg=VIEWER_FADE_DEG, clear_rgb=list(VIEWER_CLEAR_RGB),
                         rule="X-32 (owner 2026-10-26: exclude sky from 3d maps): a plain neutral brightness-by-elevation "
                              "gradient behind the splats; NOT fitted to the scene; class-D provisional constants",
                         disclosure="sky not reconstructed; plain backdrop")


class GradientBackdrop:
    """X-32: grey(e) = horizon + (zenith - horizon) * sin(e) for elevation e >= 0 against the map's up vector; below the
    horizon a smoothstep fade over below_horizon_fade_deg to the clear colour (as the viewer fades below its min_lat)."""

    def __init__(self, up, device="cuda", spec=None):
        self.spec = dict(spec or GRADIENT_BACKDROP)
        u = np.asarray(up, float)
        self.up = torch.tensor(u / np.linalg.norm(u), dtype=torch.float32, device=device)
        self.clear = torch.tensor(self.spec["clear_rgb"], dtype=torch.float32, device=device)
        self.device = device

    def sample(self, dirs):
        s = (torch.nn.functional.normalize(dirs, dim=-1) @ self.up).clamp(-1, 1)
        g = self.spec["horizon_grey"] + (self.spec["zenith_grey"] - self.spec["horizon_grey"]) * s.clamp(min=0)
        c = g[..., None].expand(*g.shape, 3)
        lat = torch.asin(s)
        t = ((0.0 - lat) / np.radians(self.spec["below_horizon_fade_deg"])).clamp(0, 1)
        k = (t * t * (3 - 2 * t))[..., None]
        return c * (1 - k) + self.clear * k


def gradient_scores(model, prep, views=None):
    """X-32 headline: held-out PSNR / SSIM of the splats composited over the plain gradient backdrop (all held-out views
    and the non-excluded ones), at the SLAM poses, identity exposure."""
    from . import sky as SKY
    bd = GradientBackdrop(prep.up, device=str(model.device))
    win = IO.gauss_win()
    views = prep.hold_ids if views is None else views
    rows = []
    for j in views:
        T = prep.T_wc(j)
        rgb0, al, _ = model.view(T, prep.K, prep.W, prep.H, with_sky=False)
        a = torch.as_tensor(al, device=model.device)[..., None]
        r0 = torch.as_tensor(rgb0, device=model.device)
        g = bd.sample(SKY.ray_dirs_world(T, prep.K, prep.W, prep.H, model.device))
        on = (r0 + (1 - a) * g).clamp(0, 1).cpu().numpy()
        img = _img(prep, j)
        with torch.no_grad():
            ss = float(IO.ssim(torch.tensor(on).permute(2, 0, 1)[None], torch.tensor(img).permute(2, 0, 1)[None], win))
        rows.append(dict(j=int(j), excluded=_is_excluded(prep.views[j]), psnr=IO.psnr_np(on, img), ssim=ss))

    def agg(sel):
        return dict(n=len(sel), psnr_db=float(np.mean([r["psnr"] for r in sel])) if sel else None,
                    ssim=float(np.mean([r["ssim"] for r in sel])) if sel else None)
    return dict(all=agg(rows), without_excluded=agg([r for r in rows if not r["excluded"]]), backdrop=bd.spec)


def viewer_scores(model, prep, md, views=None):
    """Held-out PSNR / SSIM with EXACTLY what the staging viewer draws (review3 round 2 H3/G5): the splats over the exported
    sky-only backdrop, filled and faded as sky.js (backdrop ON, the default), and the splats over the clear colour
    (backdrop OFF). At xsesto's poses, identity exposure, all held-out views and the non-excluded ones."""
    from . import sky as SKY
    bd = ViewerBackdrop(md, device=str(model.device))
    win = IO.gauss_win()
    views = prep.hold_ids if views is None else views
    rows = []
    for j in views:
        T = prep.T_wc(j)
        rgb0, al, _ = model.view(T, prep.K, prep.W, prep.H, with_sky=False)
        a = torch.as_tensor(al, device=model.device)[..., None]
        r0 = torch.as_tensor(rgb0, device=model.device)
        sky = bd.sample(SKY.ray_dirs_world(T, prep.K, prep.W, prep.H, model.device))
        on = (r0 + (1 - a) * sky).clamp(0, 1).cpu().numpy()
        off = (r0 + (1 - a) * bd.clear).clamp(0, 1).cpu().numpy()
        img = _img(prep, j)
        with torch.no_grad():
            ss = float(IO.ssim(torch.tensor(on).permute(2, 0, 1)[None], torch.tensor(img).permute(2, 0, 1)[None], win))
        rows.append(dict(j=int(j), excluded=_is_excluded(prep.views[j]), on=IO.psnr_np(on, img), off=IO.psnr_np(off, img), ssim_on=ss))

    def agg(sel):
        return dict(n=len(sel), psnr_backdrop_on_db=float(np.mean([r["on"] for r in sel])) if sel else None,
                    psnr_backdrop_off_db=float(np.mean([r["off"] for r in sel])) if sel else None,
                    ssim_backdrop_on=float(np.mean([r["ssim_on"] for r in sel])) if sel else None)
    return dict(all=agg(rows), without_excluded=agg([r for r in rows if not r["excluded"]]), backdrop_filled=bd.filled,
                min_lat_deg=bd.min_lat,
                rule=("staging_site/js/pages/viewer/sky.js ported 1:1 (fill unobserved sky per elevation, feather 6 texels, "
                      "fade over 4 deg below min_lat to the clear colour 0x04060a); splats composited over it at xsesto's "
                      "poses, identity exposure; backdrop OFF = the splats over the clear colour"))
