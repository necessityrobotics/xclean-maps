# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Per-pixel masks (BUILDER 4.3-4.6). One uint8 per pixel, bit flags as losses.BIT_*.

  valid      RAFT disparity > 0.5 px
  edge       occlusion edge, |grad d| > 1 px/px (dilated by 1 px)
  saturated  any channel >= 250/255
  sky        CONFIDENT sky -> coverage target 0 (phase 2a, review SKY-1..5): top-border connected, above the gravity
             horizon minus 2.5 deg (SKY-5: the per-frame g_w up is tilted 1-2 deg on some rows), low texture or
             brightness >= 250, AND
               colour rows: disparity < fB / 20 m AND a colour guard (blue-dominant or saturated), so a non-sky surface
                            cannot become confident sky;
               grey rows:   disparity < 0.5 px (beyond 71-83 m: RAFT says truly far). A white facade on grey OAK gets
                            1.0-1.7 px from RAFT (SKY-2), so it can never be confident sky there.
  sky_maybe  POSSIBLE sky (SKY-2/3/4): top-connected, above horizon - 2.5 deg, low texture or bright, disparity < fB / 8 m,
             and not confident. Excluded from the depth loss, the coverage positives and the seeds; NEVER target 0.
             History: BUILDER's "disparity < 0.5 px" masked 0 % of Rosario (RAFT gives 0.9-2.0 px on the blue sky); the
             phase-1 "beyond 20 m" rule leaked onto white OAK facades and missed sky RAFT put nearer than 20 m.
  moving     static-world test (OAK only): the keyframe's RAFT depth, reprojected with our poses into its neighbours,
             disagrees with their RAFT disparity in both (or the only) neighbour(s); OR a HOG person box (OAK)
  camfixed   pixels whose intensity barely changes across keyframes spread along the route (rig parts in frame); used
             only after its overlay is reviewed (5.5 INFERENCE)
  privacy    switch OFF (A4: owner "go ahead without masking anything"); privacy_blur() is a no-op that says so

XCLEAN FORK (2026-10-26, internal xclean_maps_v1): the rules above were written for RAFT (dense: a value everywhere, small
values on the sky). With depth_kind="sgm32" (our 32-direction SGM, xclean/stereo/sgm32.py: 0 = INVALID = no
measurement, never "far"; every valid value is >= 0.5 px, LR-checked and unique) they are RESTATED, not re-tuned:
  valid      disp > 0 (SGM32's own validity contract; its 0.5 px floor is MIN_DISP_PX inside SGM32)          class D
  edge       |grad d| > 1 px/px between VALID neighbours only, OR a valid pixel 4-adjacent to an invalid one (the
             disparity gradient is undefined at a hole border; LR holes sit at occlusions), dilated 1 px     class D
  sky (colour rows)  confident = top-connected, above horizon - 2.5 deg, smooth/bright, blue-or-saturated (the
             colour guard is the evidence) AND (invalid OR valid d < fB / 20 m): an invalid pixel neither proves nor
             blocks sky; a VALID nearer measurement blocks it                                             (as v4)
  sky (grey rows)    v4 needed a MEASURED d < 0.5 px, which SGM32 can never report (MIN_DISP_PX = 0.5), so the
             grey rule is restated with the colour rows' saturation clause: confident = top-connected, above horizon,
             smooth/bright, SATURATED (>= 250) AND invalid (no stereo measurement)                         class D
  sky_maybe  top-connected, above horizon, smooth/bright AND (invalid OR valid d < fB / 8 m)                (as v4)
  indoor families (config.FAMILY[...]["sky"] False, e.g. EuRoC): no sky classes at all (a class-A scene fact: there
             is no sky indoors; the rule would otherwise hit textureless ceilings and lamps that SGM32 leaves invalid)
X-33 (registered 2026-10-27 00:02 IST, CHANGES.md sha1 2cb7a0d0): MASK_PARAMS["sgm32_sky_rule"] = "x33" resolves by the
camera's sensor type (FAMILY channels, class A): grey -> "x33g", colour -> "v3" (X-30's colour resolution, unchanged).
  sky (x33g)        confident = above horizon - 2.5 deg AND GSKY-1 q >= 128 (P >= 0.5; builder/sky_prob.py, our own
             network sky_model/, trained from scratch) AND NOT measured_near, measured_near = SGM32 valid AND u > 13.93
             (X-29) AND d >= fB / 20 m. No smooth / saturated / unmeasured / top-connected terms.
  sky_maybe (x33g)  the v1 grey maybe clause UNCHANGED (top-connected, above, smooth/bright, invalid or d < fB / 8 m),
             minus confident.
"""
from __future__ import annotations

import numpy as np

from . import config as CFG
from .losses import BIT_CAMFIXED, BIT_EDGE, BIT_MOVING, BIT_SAT, BIT_SKY, BIT_SKY_MAYBE, BIT_VALID

import math as _math
# X-26 (registered 18:28): noise-only depth of the raw census cost minimum below its mean, for N_d = 128 independent costs
# between unrelated 80-bit codes (Binomial(80, 1/2), sd sqrt(80/4)): class B from SGM32's class-D census size / range
DELTA_NOISE = _math.sqrt(80 / 4.0) * _math.sqrt(2.0 * _math.log(128))
X33_Q = 128                                          # X-33: q = floor(255 P + 0.5) >= 128  <=>  P >= 0.5 (class D)
# xclean-maps: the GSKY-1 learned sky model (X-33 / X-34) is NOT shipped: it failed its registered M bars and no website
# map used it. The x33 / x33g rules therefore refuse (row_mask_params). See CHANGES_FROM_INTERNAL.md.
X33_FROZEN = None


def x33_model_sha1():
    return None


MASK_PARAMS = dict(
    sgm32_sky_rule="v1", sgm32_edge_rule="ring",       # X-26 / X-27 selection keys (defaults = the pilots' rules)
    valid_disp_px=0.5, edge_grad_px=1.0, sat_level=250, sky_far_m=20.0, sky_maybe_far_m=8.0, sky_tex_thr=3.0,
    sky_tex_win=7, sky_bright=250, sky_horizon_tol_deg=2.5, sky_blue_margin=15, sky_blue_min=110, grey_conf_disp_px=0.5,
    scan_proxy_tol_deg=3.0, scan_grey_bright=200, scan_leak_frame_max=0.01, scan_miss_frame=0.05, scan_miss_frames_max=0.05,
    sky_band_px=7, overcast_bright=180, overcast_chroma_max=20, scan_miss_median_max=0.002, overcast_unconnected=False,
    moving_disp_tol_px=1.0, moving_disp_tol_rel=0.10, moving_min_dt_s=0.2, moving_max_dt_s=1.0,
    hog_weight_min=0.5, camfixed_std_max=3.0, camfixed_min_travel_m=2.0, camfixed_n=120,
)


def grey_of(img):
    import cv2
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def saturated(img, level=250):
    if img.ndim == 2:
        return img >= level
    return (img >= level).any(-1)


def edge_mask(disp, thr=1.0):
    import cv2
    gy, gx = np.gradient(disp.astype(np.float32))
    e = np.sqrt(gx * gx + gy * gy) > thr
    return cv2.dilate(e.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def edge_mask_holes(disp, thr=1.0):
    """XCLEAN (SGM32 maps with holes, 0 = invalid): |grad d| > thr from differences between VALID neighbours only
    (central where both neighbours are valid, one-sided where one is: np.gradient's semantics), OR a valid pixel with an
    invalid 4-neighbour inside the image (hole border: the gradient is undefined there); dilated 1 px exactly as
    edge_mask. On a map without holes this equals edge_mask."""
    import cv2
    d = disp.astype(np.float32)
    v = d > 0
    l = np.zeros_like(v); r = np.zeros_like(v); u = np.zeros_like(v); dn = np.zeros_like(v)
    l[:, 1:] = v[:, :-1]; r[:, :-1] = v[:, 1:]; u[1:, :] = v[:-1, :]; dn[:-1, :] = v[1:, :]
    dl = np.zeros_like(d); dr = np.zeros_like(d); du = np.zeros_like(d); dd = np.zeros_like(d)
    dl[:, 1:] = d[:, 1:] - d[:, :-1]; dr[:, :-1] = d[:, 1:] - d[:, :-1]
    du[1:, :] = d[1:, :] - d[:-1, :]; dd[:-1, :] = d[1:, :] - d[:-1, :]
    gx = np.where(l & r, 0.5 * (dl + dr), np.where(l, dl, np.where(r, dr, 0.0)))
    gy = np.where(u & dn, 0.5 * (du + dd), np.where(u, du, np.where(dn, dd, 0.0)))
    e = v & (np.sqrt(gx * gx + gy * gy) > thr)
    nb_inv = np.zeros_like(v)
    nb_inv[:, 1:] |= ~v[:, :-1]; nb_inv[:, :-1] |= ~v[:, 1:]; nb_inv[1:, :] |= ~v[:-1, :]; nb_inv[:-1, :] |= ~v[1:, :]
    e |= v & nb_inv
    return cv2.dilate(e.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def depth_valid(disp, depth_kind="raft", p=None, scale=1.0):
    """The VALID rule of the depth source (module docstring)."""
    p = p or MASK_PARAMS
    if depth_kind == "sgm32":
        return disp > 0
    return disp > p["valid_disp_px"] * scale


def edge_mask_valid_pairs(disp, thr=1.0):
    """X-27 arm B: the valid-pair gradient part of edge_mask_holes only (no hole ring), dilated 1 px."""
    import cv2
    d = disp.astype(np.float32)
    v = d > 0
    l = np.zeros_like(v); r = np.zeros_like(v); u = np.zeros_like(v); dn = np.zeros_like(v)
    l[:, 1:] = v[:, :-1]; r[:, :-1] = v[:, 1:]; u[1:, :] = v[:-1, :]; dn[:-1, :] = v[1:, :]
    dl = np.zeros_like(d); dr = np.zeros_like(d); du = np.zeros_like(d); dd = np.zeros_like(d)
    dl[:, 1:] = d[:, 1:] - d[:, :-1]; dr[:, :-1] = d[:, 1:] - d[:, :-1]
    du[1:, :] = d[1:, :] - d[:-1, :]; dd[:-1, :] = d[1:, :] - d[:-1, :]
    gx = np.where(l & r, 0.5 * (dl + dr), np.where(l, dl, np.where(r, dr, 0.0)))
    gy = np.where(u & dn, 0.5 * (du + dd), np.where(u, du, np.where(dn, dd, 0.0)))
    e = v & (np.sqrt(gx * gx + gy * gy) > thr)
    return cv2.dilate(e.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def depth_edges(disp, depth_kind="raft", p=None):
    p = p or MASK_PARAMS
    if depth_kind == "sgm32":
        if p.get("sgm32_edge_rule", "ring") == "valid_pairs":
            return edge_mask_valid_pairs(disp, p["edge_grad_px"])
        return edge_mask_holes(disp, p["edge_grad_px"])
    return edge_mask(disp, p["edge_grad_px"])


def low_texture(grey, thr=3.0, win=7):
    import cv2
    g = grey.astype(np.float32)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = (np.abs(sx) + np.abs(sy)) / 8.0
    return cv2.blur(mag, (win, win)) < thr


def ray_elevation(T_wc, K, W, H, up):
    """sin(elevation) of every pixel ray in the world (above the gravity horizon where > 0). OpenCV pixel convention:
    pixel (i, j) has its centre at image coordinates (j, i) (review F5; the same grid as cv2.remap and the RAFT cache)."""
    jj, ii = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    d = np.stack([(jj - K[0, 2]) / K[0, 0], (ii - K[1, 2]) / K[1, 1], np.ones_like(jj)], -1)
    dw = d @ T_wc[:3, :3].T
    dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
    return dw @ (np.asarray(up) / np.linalg.norm(up))


def top_connected(cand):
    import cv2
    if not cand.any():
        return cand
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
    top = np.unique(lab[0][cand[0]])
    top = top[top > 0]
    return np.isin(lab, top) & cand


def connected_to(cand, seed):
    """Components of cand (4-connected) that touch the top border or overlap seed."""
    import cv2
    if not cand.any():
        return cand
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
    keep = np.unique(np.concatenate([lab[0][cand[0]], lab[seed & cand]]))
    keep = keep[keep > 0]
    return np.isin(lab, keep) & cand


def is_colour(img):
    return img.ndim == 3 and img.shape[2] == 3 and not (np.array_equal(img[..., 0], img[..., 1]) and
                                                        np.array_equal(img[..., 1], img[..., 2]))


def blue_or_saturated(img, p=MASK_PARAMS):
    """Colour guard (and the scan's sky proxy): blue-dominant (b > r + 15, b >= g, b > 110) or any channel saturated."""
    r, g, b = [img[..., i].astype(np.int16) for i in range(3)]
    blue = (b > r + p["sky_blue_margin"]) & (b >= g) & (b > p["sky_blue_min"])
    return blue | (img >= p["sat_level"]).any(-1)


def row_mask_params(spec):
    """MASK_PARAMS with the build's mask_overrides (spec: dict(family=..., mask_overrides=...)) (e.g. the overcast forest: overcast_unconnected)."""
    p = dict(MASK_PARAMS)
    p.update(spec.get("mask_overrides") or {})
    if p.get("sgm32_sky_rule") == "x30":
        # X-30 (registered 22:01, post hoc): by the camera's sensor type, a class-A fact of the family table
        ch = CFG.FAMILY[spec["family"]]["channels"]
        p["sgm32_sky_rule"] = "v3" if ch == 3 else "v1"
        p["sgm32_sky_rule_x30"] = "x30 -> %s (FAMILY[%s] channels %d: %s)" % (p["sgm32_sky_rule"], spec["family"], ch,
                                                                          "colour" if ch == 3 else "grey")
    if p.get("sgm32_sky_rule") in ("x33", "x33g"):
        raise SystemExit("sky rule %s needs the GSKY-1 sky model, which xclean-maps does not ship (it failed its "
                         "registered tests; no website map used it)" % p.get("sgm32_sky_rule"))
    if p.get("sgm32_sky_rule") == "x33":
        # X-33 (registered 2026-10-27 00:02): by the camera's sensor type, a class-A fact of the family table
        ch = CFG.FAMILY[spec["family"]]["channels"]
        p["sgm32_sky_rule"] = "x33g" if ch == 1 else "v3"
        p["sgm32_sky_rule_x33"] = "x33 -> %s (FAMILY[%s] channels %d: %s)" % (p["sgm32_sky_rule"], spec["family"], ch,
                                                                          "grey" if ch == 1 else "colour")
        if ch == 1:
            p["x33_model_sha1"] = x33_model_sha1()
    return p


def overcast_like(img, p=MASK_PARAMS):
    """Bright, low-chroma, low-texture pixels (an overcast grey-white sky; colour rows only). Review SKY-NEW-2."""
    g = grey_of(img)
    ch = img.max(-1).astype(np.int16) - img.min(-1).astype(np.int16)
    return (g >= p["overcast_bright"]) & (ch < p["overcast_chroma_max"]) & low_texture(g, p["sky_tex_thr"], p["sky_tex_win"])


def _dilate(m, r):
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def sky_classes(img, disp, T_wc, K, up, fB, p=MASK_PARAMS, scale=1.0, el=None, depth_kind="raft", delta=None, skyq=None):
    """-> (confident_sky, maybe_sky) boolean maps (see the module docstring).
    Phase 2b additions to MAYBE (colour rows): (SKY-NEW-1) the sky band next to confident sky that the texture window
    misses (within sky_band_px of confident sky, blue-or-saturated, above the horizon, beyond fB / sky_maybe_far_m);
    (SKY-NEW-2) an overcast sky: top-connected bright low-chroma low-texture pixels above the horizon, with NO disparity
    bound (RAFT smears branch disparity into it). Both only EXCLUDE pixels (depth loss, coverage positives, seeds)."""
    H, W = disp.shape
    grey = grey_of(img)
    el = ray_elevation(T_wc, K, W, H, up) if el is None else el
    above = el > -np.sin(np.radians(p["sky_horizon_tol_deg"]))
    smooth = low_texture(grey, p["sky_tex_thr"], p["sky_tex_win"]) | (grey >= p["sky_bright"])
    base = above & smooth
    colour = is_colour(img)
    if depth_kind == "sgm32" and p.get("sgm32_sky_rule") == "x33g":
        # X-33 (registered 2026-10-27 00:02): grey rows; GSKY-1 probability q; the maybe clause is v1's, unchanged
        if colour:
            raise ValueError("x33g is the grey-row resolution of X-33; this image is colour")
        if skyq is None or delta is None:
            raise ValueError("sky rule x33g needs the sky stage's q array and the X-29 evidence u")
        inv = disp <= 0
        d0 = np.where(inv, 0.0, disp)
        far8 = inv | (d0 < fB / p["sky_maybe_far_m"])
        near = (disp > 0) & (delta > DELTA_NOISE) & (disp >= fB / p["sky_far_m"])
        conf = above & (np.asarray(skyq) >= X33_Q) & ~near
        maybe = top_connected(base & far8)
        return conf, maybe & ~conf
    if depth_kind == "sgm32":
        if p.get("sgm32_sky_rule", "v1") in ("v2", "v3"):
            # X-26 (v2): delta = mean - min of the raw census cost; X-29 (v3): delta = u, the raw non-adjacent second
            # minimum - the raw minimum. Either way a valid SGM32 value whose own raw data term does not single out a
            # disparity by more than the class-B band (13.93) is a smoothness propagation: unmeasured (sky tests only)
            if delta is None:
                raise ValueError("sky rule %s needs its evidence array (v2: delta, v3: u)" % p.get("sgm32_sky_rule"))
            inv = ~((disp > 0) & (delta > DELTA_NOISE))
        else:
            inv = disp <= 0                              # no measurement: neither proves nor blocks sky
        disp = np.where(inv, 0.0, disp)                  # unmeasured pixels never satisfy a 'nearer than' test
        far20 = inv | (disp < fB / p["sky_far_m"])
        far8 = inv | (disp < fB / p["sky_maybe_far_m"])
        if colour:
            conf = base & far20 & blue_or_saturated(img, p)
        else:
            conf = base & inv & saturated(img, p["sat_level"])
    else:
        far8 = disp < fB / p["sky_maybe_far_m"]
        if colour:
            conf = base & (disp < fB / p["sky_far_m"]) & blue_or_saturated(img, p)
        else:
            conf = base & (disp < p["grey_conf_disp_px"] * scale)
    conf = top_connected(conf)
    maybe = top_connected(base & far8)
    if colour:
        r = max(2, int(round(p["sky_band_px"] * scale)))
        band = _dilate(conf, r) & blue_or_saturated(img, p) & above & far8
        maybe |= band
        open_sky = top_connected(overcast_like(img, p) & above)
        maybe |= open_sky
        # sky seen through bare branches (SKY-NEW-2): bright low-chroma pixels above the horizon, connected (through such
        # pixels) to the top border or to the open overcast sky; no texture test (the branches make every window busy)
        ch = img.max(-1).astype(np.int16) - img.min(-1).astype(np.int16)
        cand = (grey >= p["overcast_bright"]) & (ch < p["overcast_chroma_max"]) & above
        maybe |= cand if p.get("overcast_unconnected") else connected_to(cand, open_sky)
    return conf, maybe & ~conf


def sky_mask(img, disp, T_wc, K, up, p=MASK_PARAMS, fB=None, scale=1.0):
    """Confident sky only (kept for callers that need one map)."""
    return sky_classes(img, disp, T_wc, K, up, fB if fB else 10.0, p, scale)[0]


def basic_bits(img, disp, T_wc, K, up, sky=True, p=MASK_PARAMS, fB=None, scale=1.0, depth_kind="raft", delta=None,
               skyq=None):
    """valid | edge | saturated | sky | sky_maybe bits for one keyframe (disp may be None: no depth -> only saturated).
    scale = training-resolution scale (rosariov2_01: 0.5): pixel-based disparity thresholds scale with it.
    depth_kind: "raft" (the v4 rules, unchanged) or "sgm32" (the xclean restatement in the module docstring)."""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    m |= np.where(saturated(img, p["sat_level"]), BIT_SAT, 0).astype(np.uint8)
    if disp is None:
        return m
    m |= np.where(depth_valid(disp, depth_kind, p, scale), BIT_VALID, 0).astype(np.uint8)
    m |= np.where(depth_edges(disp, depth_kind, p), BIT_EDGE, 0).astype(np.uint8)
    if sky and up is not None and fB:
        conf, maybe = sky_classes(img, disp, T_wc, K, up, fB, p, scale, depth_kind=depth_kind, delta=delta, skyq=skyq)
        m |= np.where(conf, BIT_SKY, 0).astype(np.uint8)
        m |= np.where(maybe, BIT_SKY_MAYBE, 0).astype(np.uint8)
    return m


def proxy_sky(img, above, p=MASK_PARAMS):
    """The scan's 'true sky' proxy: colour rows blue-dominant, saturated OR overcast-like (SKY-NEW-2: the phase-2a
    proxy was blind to grey-white skies); grey rows bright and low texture; top-connected, above the horizon."""
    if is_colour(img):
        prox = blue_or_saturated(img, p) | overcast_like(img, p)
    else:
        g = grey_of(img)
        prox = (g >= p["scan_grey_bright"]) & low_texture(g, p["sky_tex_thr"], p["sky_tex_win"])
    return top_connected(prox & above)


def sky_scan_frame(img, bits, T_wc, K, up, p=MASK_PARAMS):
    """Route-wide missed-sky / leak scan, one frame (review SKY-3/4; the reviewer's route_scan.py proxy).
    Proxy of 'true' sky: colour rows blue-dominant or saturated, grey rows bright (>= 200) and low texture; top-connected,
    above the horizon - 3 deg. missed = proxy and not (sky or maybe); missed_valid_notsat = the part that would enter the
    depth loss / coverage positives; leak = confident sky that is NOT proxy sky (a surface masked as sky)."""
    H, W = bits.shape
    el = ray_elevation(T_wc, K, W, H, up)
    above = el > -np.sin(np.radians(p["scan_proxy_tol_deg"]))
    prox = proxy_sky(img, above, p)
    sky = (bits & BIT_SKY) != 0
    maybe = (bits & BIT_SKY_MAYBE) != 0
    miss = prox & ~sky & ~maybe
    mvn = miss & ((bits & BIT_VALID) != 0) & ((bits & BIT_SAT) == 0)
    return dict(sky=float(sky.mean()), maybe=float(maybe.mean()), proxy=float(prox.mean()), missed=float(miss.mean()),
                missed_valid_notsat=float(mvn.mean()), leak=float((sky & ~prox).mean()),
                proxy_kind="colour (blue / saturated / overcast)" if is_colour(img) else "grey")


def _ranges(ts, gap=2.0):
    if not ts:
        return []
    out, a, b = [], ts[0], ts[0]
    for t in ts[1:]:
        if t - b > gap:
            out.append([round(a, 1), round(b, 1)])
            a = t
        b = t
    out.append([round(a, 1), round(b, 1)])
    return out


def sky_scan_verdict(frames, times, p=MASK_PARAMS):
    """Row decision, pre-registered in THRESHOLDS.md (phase 2a): FAIL if any keyframe has leak > 1 % of the frame, or if
    more than 5 % of keyframes have missed-and-valid-unsaturated sky > 5 % of the frame."""
    leak = np.array([f["leak"] for f in frames])
    mvn = np.array([f["missed_valid_notsat"] for f in frames])
    bad_leak = np.nonzero(leak > p["scan_leak_frame_max"])[0]
    bad_miss = np.nonzero(mvn > p["scan_miss_frame"])[0]
    frac_miss = float(len(bad_miss) / max(len(frames), 1))
    med_mvn = float(np.median(mvn)) if len(mvn) else 0.0
    ok = (len(bad_leak) == 0) and (frac_miss <= p["scan_miss_frames_max"]) and (med_mvn <= p["scan_miss_median_max"])
    q = lambda k: [float(np.percentile([f[k] for f in frames], x)) for x in (10, 50, 90)] if frames else None  # noqa: E731
    return {"pass": bool(ok), "n_frames": len(frames), "n_leak_frames": int(len(bad_leak)),
            "leak_max": float(leak.max()) if len(leak) else 0.0, "leak_ranges_s": _ranges([float(times[i]) for i in bad_leak]),
            "n_missed_valid_frames": int(len(bad_miss)), "missed_valid_frame_share": frac_miss,
            "missed_valid_ranges_s": _ranges([float(times[i]) for i in bad_miss]),
            "sky_share_p10_50_90": q("sky"), "maybe_share_p10_50_90": q("maybe"), "missed_share_p10_50_90": q("missed"),
            "missed_valid_median": med_mvn,
            "rule": "FAIL if any keyframe leak > 1 % of the frame, or > 5 % of keyframes with missed valid unsaturated sky > 5 %, "
                    "or the route median of missed valid unsaturated sky > 0.2 % of the frame (phase 2b, SKY-NEW-1)"}


def reproject_disparity(disp_a, T_wa, T_wb, K, fB, stride=1):
    """Pixels of view a (valid disparity) -> view b: (ub, vb, predicted disparity in b, valid) on a stride grid."""
    H, W = disp_a.shape
    jj, ii = np.meshgrid(np.arange(0, W, stride) + 0.0, np.arange(0, H, stride) + 0.0)
    d = disp_a[::stride, ::stride]
    ok = d > 0.5
    z = np.where(ok, fB / np.maximum(d, 1e-6), 0.0)
    X = np.stack([(jj - K[0, 2]) / K[0, 0] * z, (ii - K[1, 2]) / K[1, 1] * z, z], -1)
    T_ba = np.linalg.inv(T_wb) @ T_wa
    Xb = X @ T_ba[:3, :3].T + T_ba[:3, 3]
    zb = Xb[..., 2]
    ok &= zb > 0.05
    ub = K[0, 0] * Xb[..., 0] / np.maximum(zb, 1e-6) + K[0, 2]
    vb = K[1, 1] * Xb[..., 1] / np.maximum(zb, 1e-6) + K[1, 2]
    ok &= (ub >= 0) & (ub <= W - 1) & (vb >= 0) & (vb <= H - 1)
    return ub, vb, np.where(ok, fB / np.maximum(zb, 1e-6), 0.0), ok


def moving_mask(disp_a, T_wa, neighbours, K, fB, p=MASK_PARAMS):
    """neighbours: list of (disp_b, T_wb). A pixel is MOVING when every neighbour that sees it disagrees."""
    H, W = disp_a.shape
    bad_all = np.ones((H, W), bool)
    seen_any = np.zeros((H, W), bool)
    for disp_b, T_wb in neighbours:
        ub, vb, dpred, ok = reproject_disparity(disp_a, T_wa, T_wb, K, fB)
        ui, vi = np.clip(np.rint(ub).astype(int), 0, W - 1), np.clip(np.rint(vb).astype(int), 0, H - 1)
        db = disp_b[vi, ui]
        ok &= db > 0.5
        tol = np.maximum(p["moving_disp_tol_px"], p["moving_disp_tol_rel"] * dpred)
        disagree = ok & (np.abs(dpred - db) > tol)
        bad_all &= ~(ok & ~disagree)            # any agreeing neighbour clears the pixel
        seen_any |= ok
    return bad_all & seen_any


def hog_people(grey, weight_min=0.5):
    """OpenCV's built-in HOG people detector (no model file); boxes as a mask."""
    import cv2
    if not hasattr(cv2, "HOGDescriptor"):
        raise SystemExit("xclean-maps: this OpenCV (%s) has no HOGDescriptor (the people mask needs it). OpenCV 5 keeps "
                         "HOG in the contrib build: pip uninstall -y opencv-python && pip install "
                         "opencv-contrib-python==%s" % (cv2.__version__, cv2.__version__))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(grey, winStride=(8, 8), padding=(8, 8), scale=1.05)
    m = np.zeros(grey.shape[:2], bool)
    for (x, y, w, h), wt in zip(rects, np.asarray(weights).reshape(-1)):
        if float(wt) >= weight_min:
            m[max(0, y):y + h, max(0, x):x + w] = True
    return m


def camera_fixed(greys, p=MASK_PARAMS):
    """greys: list of grey keyframes spread along the route. Pixels with a tiny temporal std are rig parts."""
    import cv2
    a = np.stack([g.astype(np.float32) for g in greys], 0)
    sd = a.std(0)
    m = sd < p["camfixed_std_max"]
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    return m, sd


def privacy_blur(images, enabled=CFG.PRIVACY_BLUR_ENABLED):
    """No privacy blur is applied (the internal builder had the same switch OFF). No-op by design."""
    if enabled:
        raise RuntimeError("xclean-maps has no privacy blur")
    return images, dict(enabled=False, reason=CFG.PRIVACY_NOTE)


__all__ = ["MASK_PARAMS", "basic_bits", "sky_mask", "moving_mask", "hog_people", "camera_fixed", "privacy_blur",
           "BIT_CAMFIXED", "BIT_MOVING"]
