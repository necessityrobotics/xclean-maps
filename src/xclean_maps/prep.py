# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""DEPTH + PREP: the training inputs of one map, from an xclean output folder + the sequence it ran on.

  depth  (GPU, CuPy)  <out>/cache/depth/depth_full.npy [n, H, W] float32 SGM32 disparity at full resolution
                      (0 = INVALID), depth_index.json (frames, per-frame sha1, SGM32's contract, xclean module hashes)
  prep   (CPU)        <out>/cache/{img,disp,mask}.npy, seeds.npz, views.json; <out>/work/plan.json, sky_scan.json,
                      <out>/work/poses_used/{trajectory_kf.csv, trajectory_frames.csv}

Views = the run's keyframes (runs.py). Poses = the run's own, unchanged. Images = the sequence's left camera,
rectified with xclean's own rectification maps. Seeds = dense depth only. Hold-out every 10th view, frame quality,
masks, sky scan, scale L, budget, steps and voxel de-duplication are the internal builder's code (prep_xclean.py +
prep_row.py), unchanged in their maths; see CHANGES_FROM_INTERNAL.md for what was removed.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import time
from types import SimpleNamespace

import numpy as np

from . import config as CFG
from . import frame_quality as FQ
from . import io_lib as IO
from . import masks as MK
from . import runs as RUNS
from . import scale as SC
from . import sources as SRC
from . import util as U
from .losses import BIT_CAMFIXED, BIT_EDGE, BIT_MOVING, BIT_SAT, BIT_SKY, BIT_SKY_MAYBE, BIT_VALID

PREP_VERSION = "prep-xclean-maps-1 (= internal prep-xclean-1 maths)"
DEPTH_VERSION = "depth-xclean-1"
# FQ-2 restated for SGM32 (internal CHANGES X-07, class D): a frame seeds when it has ANY valid unsaturated non-edge pixel
SEED_VALID_MIN = {"sgm32": 0.0}


def select_views(run, limit=None):
    frames = run.kf_frames
    if not frames:
        raise SystemExit("%s: the run has no keyframe list (p1_frames.jsonl or kit/trajectory_kf.csv)" % run.dir)
    frames = list(frames)
    if limit:
        pick = np.unique(np.round(np.linspace(0, len(frames) - 1, int(limit))).astype(int))
        frames = [frames[i] for i in pick]
    return frames


def depth_paths(dirs):
    d = dirs["hot"] + "/depth"
    return dict(dir=d, arr=d + "/depth_full.npy", index=d + "/depth_index.json", delta=d + "/delta_full.npy",
                uniq=d + "/uniq_full.npy")


def holdout_roles(n, every=10, old_hold_frames=None, frame_index=None, probe_n=64):
    """every 10th keyframe by index; with an old map: its held-out PHOTOS that are v4 keyframes, and v4's own pick is
    dropped where an old held-out photo already sits within its +-5 keyframes (keeps ~10 % per stretch, P 4.1)."""
    hold = np.zeros(n, bool)
    hold[np.arange(0, n, every)] = True
    same_photo = np.zeros(n, bool)
    if old_hold_frames is not None and frame_index is not None:
        same_photo = np.isin(frame_index, np.asarray(sorted(old_hold_frames)))
        if same_photo.any():
            near = np.convolve(same_photo.astype(int), np.ones(2 * (every // 2) + 1, int), mode="same") > 0
            hold = (hold & ~near) | same_photo
    hi = np.nonzero(hold)[0]
    probe = np.zeros(n, bool)
    if len(hi):
        pick = np.unique(np.round(np.linspace(0, len(hi) - 1, min(probe_n, len(hi)))).astype(int))
        probe[hi[pick]] = True
    return hold, probe, same_photo


def scale_K(K, s):
    """Camera matrix of an image resized by s (OpenCV pixel convention: c' = (c + 0.5) s - 0.5; review F4)."""
    K2 = np.array(K, float).copy()
    if s == 1.0:
        return K2
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] = (K2[0, 2] + 0.5) * s - 0.5
    K2[1, 2] = (K2[1, 2] + 0.5) * s - 0.5
    return K2


def downscale(img, disp, Ws, Hs, s):
    """Image: INTER_AREA. Disparity: validity-weighted area average, multiplied by s (pixels shrink, so with fB * s the
    depth fB / d is unchanged)."""
    import cv2
    img2 = cv2.resize(img, (Ws, Hs), interpolation=cv2.INTER_AREA)
    if img.ndim == 3 and img2.ndim == 2:
        img2 = img2[..., None]
    if disp is None:
        return img2, None
    v = (disp > 0).astype(np.float32)
    num = cv2.resize(disp * v, (Ws, Hs), interpolation=cv2.INTER_AREA)
    den = cv2.resize(v, (Ws, Hs), interpolation=cv2.INTER_AREA)
    d2 = np.where(den >= 0.5, s * num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    return np.ascontiguousarray(img2), d2


def _npz_bytes(**arrs):
    import io
    b = io.BytesIO()
    np.savez(b, **arrs)
    return b.getvalue()


def _camera_fixed(mm_img, Twc, channels, dirs, apply, mm_mask, log):
    import cv2
    P = np.array([T[:3, 3] for T in Twc])
    pick, last = [], None
    for j in range(len(P)):
        if last is None or np.linalg.norm(P[j] - P[last]) >= MK.MASK_PARAMS["camfixed_min_travel_m"]:
            pick.append(j)
            last = j
    pick = pick[:: max(1, len(pick) // MK.MASK_PARAMS["camfixed_n"])]
    greys = [np.asarray(mm_img[j])[..., 0] if channels == 1 else MK.grey_of(np.asarray(mm_img[j])) for j in pick]
    m, sd = MK.camera_fixed(greys)
    ov = cv2.cvtColor(greys[len(greys) // 2], cv2.COLOR_GRAY2BGR)
    ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    U.atomic_write_bytes(os.path.join(dirs["work"], "camera_fixed_overlay.jpg"), cv2.imencode(".jpg", ov)[1].tobytes())
    if apply:
        for j in range(mm_mask.shape[0]):
            mm_mask[j] = np.asarray(mm_mask[j]) | np.where(m, BIT_CAMFIXED, 0).astype(np.uint8)
    log("camera-fixed mask: %.2f %% of pixels from %d keyframes (%s)" % (100 * m.mean(), len(pick),
                                                                       "APPLIED" if apply else "overlay only, not applied until reviewed"))
    return dict(share=float(m.mean()), n_keyframes=len(pick), applied=bool(apply), overlay="camera_fixed_overlay.jpg")


def build_seeds(kit, kf_ids, Twc, K, W, H, mm_img, mm_disp, mm_mask, train, seedable, rec, cap, channels, log, scale=1.0):
    """Kit landmarks first (colour from a TRAINING view: the host if it trains, else the previous training keyframe),
    then RAFT seeds from every 4th seedable training view at an 8-px stride, near clip .. min(60 m, fB / 2 px);
    depth-adaptive voxel de-duplication (max(5 cm, 0.5 % depth)); capped at 0.5 x cap_max by growing the voxel ladder
    (never a random per-region subsample); initial isotropic scale 0.1 x kNN RMS distance (gsplat MCMC preset)."""
    from scipy.spatial import cKDTree
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fB = kit.fB * scale
    pos = {int(k): j for j, k in enumerate(kf_ids)}
    train_idx = np.nonzero(train)[0]
    P, host = kit.landmarks, kit.host
    lm_xyz, lm_rgb, lm_depth = [], [], []
    for k in np.unique(host):
        j = pos.get(int(k))
        if j is None:
            continue
        if not train[j]:
            prev = train_idx[train_idx < j]
            if not len(prev):
                continue
            j = int(prev[-1])
        sel = np.nonzero(host == k)[0]
        Tcw = IO.inv_T(Twc[j])
        Pc = P[sel] @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = Pc[:, 2]
        ok = z > 1e-3
        u = np.rint(fx * Pc[:, 0] / np.maximum(z, 1e-6) + cx).astype(int)
        v = np.rint(fy * Pc[:, 1] / np.maximum(z, 1e-6) + cy).astype(int)
        inside = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        img = np.asarray(mm_img[j])
        col = np.full((len(sel), 3), 0.5, np.float32)
        c = img[v[inside], u[inside]].astype(np.float32) / 255.0
        col[inside] = c if channels == 3 else np.repeat(c, 3, axis=1)
        lm_xyz.append(P[sel])
        lm_rgb.append(col)
        lm_depth.append(np.where(ok, z, 1.0))
    lm_xyz = np.concatenate(lm_xyz) if lm_xyz else np.zeros((0, 3))
    lm_rgb = np.concatenate(lm_rgb) if lm_rgb else np.zeros((0, 3), np.float32)
    lm_depth = np.concatenate(lm_depth) if lm_depth else np.zeros(0)
    zmax = min(rec["seed_max_depth_m"], kit.fB / rec["seed_min_disp_px"])     # full-resolution pixel rule (2 px)
    stride = max(1, int(round(rec["seed_stride"] * scale)))
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    bad = BIT_SKY | BIT_SKY_MAYBE | BIT_SAT | BIT_EDGE | BIT_MOVING | BIT_CAMFIXED
    rx, rc, rz = [], [], []
    seed_views = np.nonzero(seedable)[0][::rec["seed_every"]]
    for j in seed_views:
        d = np.asarray(mm_disp[j], np.float32)[yy, xx]
        mb = np.asarray(mm_mask[j])[yy, xx]
        ok = ((mb & BIT_VALID) != 0) & ((mb & bad) == 0)
        z = np.where(ok, fB / np.maximum(d, 1e-6), 0.0)
        ok &= (z >= kit.near) & (z <= zmax)
        if not ok.any():
            continue
        X = (xx[ok] - cx) / fx * z[ok]
        Y = (yy[ok] - cy) / fy * z[ok]
        Pw = np.stack([X, Y, z[ok]], 1) @ Twc[j][:3, :3].T + Twc[j][:3, 3]
        img = np.asarray(mm_img[j])
        c = img[yy[ok], xx[ok]].astype(np.float32) / 255.0
        rx.append(Pw)
        rc.append(c if channels == 3 else np.repeat(c, 3, axis=1))
        rz.append(z[ok])
    rx = np.concatenate(rx) if rx else np.zeros((0, 3))
    rc = np.concatenate(rc) if rc else np.zeros((0, 3), np.float32)
    rz = np.concatenate(rz) if rz else np.zeros(0)
    allx = np.concatenate([lm_xyz, rx]).astype(np.float64)
    allc = np.concatenate([lm_rgb, rc]).astype(np.float32)
    allz = np.concatenate([lm_depth, rz])
    src = np.r_[np.zeros(len(lm_xyz), np.uint8), np.ones(len(rx), np.uint8)]
    limit = int(rec["seed_cap_frac"] * cap)
    grow = 1.0
    while True:
        vox = np.maximum(rec["seed_voxel_min_m"], rec["seed_voxel_frac"] * allz)       # depth-adaptive voxel
        lvl = np.ceil(np.log(vox / rec["seed_voxel_min_m"]) / np.log(1.1) - 1e-9).astype(np.int64)   # 10 % ladder
        size = rec["seed_voxel_min_m"] * 1.1 ** lvl * grow
        key = np.c_[lvl, np.floor(allx / size[:, None]).astype(np.int64)]
        _, first = np.unique(key, axis=0, return_index=True)
        first = np.sort(first)                                   # keeps landmarks first (they come first in order)
        if len(first) <= limit or grow > 1e3:
            break
        grow *= 1.25
    X, Cc, Sg = allx[first].astype(np.float32), allc[first], src[first]
    tree = cKDTree(X)
    k = min(4, len(X))
    d, _ = tree.query(X, k=k)
    dd = d[:, 1:] if k > 1 else np.full((len(X), 1), 0.01)
    rms = np.sqrt(np.mean(dd ** 2, 1))
    logs = np.log(np.clip(rms * rec["init_scale_knn"], 1e-4, None)).astype(np.float32)
    info = dict(n_landmarks_in=int(len(lm_xyz)), n_raft_in=int(len(rx)), n_seed_views=int(len(seed_views)),
                n_total=int(len(X)), n_landmarks=int((Sg == 0).sum()), n_raft=int((Sg == 1).sum()), cap=limit,
                voxel_grow=float(grow), seed_max_depth_m=float(zmax),
                rule="landmarks first + RAFT seeds every %d-th seedable training view, stride %d px, %.2f..%.1f m; voxel "
                     "max(5 cm, 0.5 %% depth) x %.3f; cap 0.5 x cap_max" % (rec["seed_every"], stride, kit.near, zmax, grow))
    log("seeds: %d landmarks + %d dense-depth -> %d after voxel de-dup (grow %.3f, cap %d)" % (len(lm_xyz), len(rx), len(X), grow, limit))
    return dict(xyz=X, rgb=Cc.astype(np.float32), log_scale=logs, source=Sg), info


def compute_depth(spec, dirs, log=print, limit=None):
    """The depth stage: our 32-direction SGM on every view (GPU, CuPy)."""
    row = spec["name"]
    t0 = time.time()
    run = RUNS.open_run(spec["run_dir"])
    ds = SRC.open_dataset(spec["sequence"], spec.get("ext"), gate=spec.get("gate", True))
    rig = SRC.rig_of(ds)
    frames = select_views(run, limit)
    P = depth_paths(dirs)
    U.remove_if_exists(P["dir"])                                  # stale-artefact rule
    U.ensure_dir(P["dir"])
    n, H, W = len(frames), rig["H"], rig["W"]
    src = SRC.Sgm32Source()
    tmp = P["arr"].replace(".npy", ".building.npy")
    mm = np.lib.format.open_memmap(U.assert_writable(tmp), mode="w+", dtype=np.float32, shape=(n, H, W))
    # X-26 / X-29 evidence arrays: only the non-default sky rules v2 / v3 read them, and the reported-only X-26.6 leak
    # indicator. xclean-maps writes them only when asked (spec depth_evidence) or when the sky rule needs them.
    want_delta = bool(spec.get("depth_evidence")) or MK.row_mask_params(spec).get("sgm32_sky_rule") in ("v2", "v3")
    tmpd = P["delta"].replace(".npy", ".building.npy")
    mmd = np.lib.format.open_memmap(U.assert_writable(tmpd), mode="w+", dtype=np.float16, shape=(n, H, W)) if want_delta else None
    tmpu = P["uniq"].replace(".npy", ".building.npy")
    mmu = np.lib.format.open_memmap(U.assert_writable(tmpu), mode="w+", dtype=np.uint8, shape=(n, H, W)) if want_delta else None
    has, sha, vshare = np.zeros(n, bool), [None] * n, np.zeros(n)
    dsf = ds.frames()
    for k, f in enumerate(frames):
        fr = dsf[f]
        if fr.index != f or abs(fr.t - run.t[f]) > 1e-6:
            raise SystemExit("frame %d: dataset index %d / t %.6f vs the run's t %.6f: the run was not made from this "
                             "sequence folder (or its frame indexing differs)" % (f, fr.index, fr.t, run.t[f]))
        d = src.disparity(ds, f)
        if d is None:
            mm[k] = 0.0
            continue
        if d.shape != (H, W):
            raise SystemExit("frame %d: disparity %s != image %s" % (f, d.shape, (H, W)))
        mm[k] = d
        if want_delta:
            mmd[k] = src.distinctiveness()
            mmu[k] = src.uniqueness_margin()                # X-29
        has[k] = True
        sha[k] = U.sha1_bytes(np.ascontiguousarray(d).tobytes())
        vshare[k] = float((d > 0).mean())
        if (k + 1) % 100 == 0:
            log("depth (%s): %d / %d views" % (src.kind, k + 1, n))
    mm.flush()
    if getattr(mm, "_mmap", None) is not None:
        mm._mmap.close()
    del mm
    import gc
    gc.collect()
    U._replace_retry(tmp, P["arr"])
    if want_delta:
        mmd.flush()
        if getattr(mmd, "_mmap", None) is not None:
            mmd._mmap.close()
        del mmd
        gc.collect()
        U._replace_retry(tmpd, P["delta"])
        mmu.flush()
        if getattr(mmu, "_mmap", None) is not None:
            mmu._mmap.close()
        del mmu
        gc.collect()
        U._replace_retry(tmpu, P["uniq"])
    idx = dict(delta=(dict(file=P["delta"], sha1=U.sha1_file(P["delta"]), rule="X-26: mean_d C - min_d C of SGM32's raw census "
                           "cost volume over in-range d", noise_level=MK.DELTA_NOISE) if want_delta else None),
               uniq=(dict(file=P["uniq"], sha1=U.sha1_file(P["uniq"]), rule="X-29: raw non-adjacent second minimum - raw "
                          "minimum over in-range d; undefined -> 0", band=MK.DELTA_NOISE) if want_delta else None),
               row=row, version=DEPTH_VERSION, depth_source=src.kind, licence_clean=src.licence_clean, contract=src.contract,
               frames=[int(f) for f in frames], has=has.tolist(), sha1=sha, valid_share=[round(float(x), 5) for x in vshare],
               valid_share_stats=U.stats(vshare[has].tolist()) if has.any() else None, stats=src.stats(),
               image_size=[W, H], rect_digest=rig["rect_digest"], fB=rig["fB"], run_dir=run.dir,
               xclean_modules=SRC.xclean_module_hashes(), seconds=round(time.time() - t0, 1),
               array_sha1=U.sha1_file(P["arr"]))
    U.atomic_json(P["index"], idx)
    log("depth (%s) done: %d views (%d with depth), valid share median %.3f, %.0f s" % (
        src.kind, n, int(has.sum()), float(np.median(vshare[has])) if has.any() else 0.0, time.time() - t0))
    return idx


def _view_job(j, ds, f, T_wc, rig, depth_arr_path, k, channels, up, sky_on, kind, sc, K, W, H, mp, delta_path=None,
              skyq_path=None):
    img, raw_sha, grey = SRC.load_left_colour(ds, f, rig["m1"])
    if channels == 1 and img.ndim == 3:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if channels == 3 and img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    arr = np.load(depth_arr_path, mmap_mode="r")
    disp = np.array(arr[k], np.float32)
    has = bool((disp > 0).any())
    delta = None
    if delta_path and os.path.isfile(delta_path):
        delta = np.array(np.load(delta_path, mmap_mode="r")[k], np.float32)
    if mp.get("sgm32_sky_rule", "v1") in ("v2", "v3", "x33g") and kind == "sgm32" and delta is None:
        raise SystemExit("sky rule %s needs the depth stage's evidence file %s (re-run the depth stage)"
                         % (mp.get("sgm32_sky_rule"), delta_path))
    fB = rig["fB"]
    Kf = rig["K"]
    if sc != 1.0:
        img, disp = downscale(img, disp, W, H, sc)
        if channels == 1 and img.ndim == 3:
            img = img[..., 0]
        Kf, fB = K, fB * sc
        if delta is not None:
            import cv2
            delta = cv2.resize(delta, (W, H), interpolation=cv2.INTER_AREA)
    skyq = None
    if mp.get("sgm32_sky_rule") == "x33g" and sky_on and kind == "sgm32":
        skyq = np.array(np.load(skyq_path, mmap_mode="r")[k], np.uint8)            # X-33 sky stage (builder/sky_prob.py)
        if skyq.shape != img.shape[:2]:
            raise SystemExit("x33g: sky stage array %s != the PREP image %s" % (skyq.shape, img.shape[:2]))
    bits = MK.basic_bits(img, disp, T_wc, Kf, up, sky=sky_on, fB=fB, scale=sc, p=mp, depth_kind=kind, delta=delta,
                         skyq=skyq)
    okm = ((bits & BIT_VALID) != 0) & ((bits & (BIT_SAT | BIT_EDGE)) == 0)
    q = FQ.measures(img, disp, scale=sc, ok_mask=okm)
    scan = MK.sky_scan_frame(img, bits, T_wc, Kf, up, p=mp) if (sky_on and up is not None) else None
    if scan is not None and skyq is not None:
        # X-33 reported only: the model-sky share (above & q >= 128) and the same-frame veto share (model sky that a
        # unique stereo match nearer than 20 m contradicts: a target-domain false-sky indicator)
        el_ = MK.ray_elevation(T_wc, Kf, img.shape[1], img.shape[0], up)
        ms_ = (el_ > -np.sin(np.radians(mp["sky_horizon_tol_deg"]))) & (skyq >= MK.X33_Q)
        near_ = (disp > 0) & (delta > MK.DELTA_NOISE) & (disp >= fB / mp["sky_far_m"])
        scan.update(x33_model_sky=float(ms_.mean()), x33_veto=float((ms_ & near_).mean()))
    return j, img, disp, bits, q, raw_sha, has, scan, grey, delta


def prepare(spec, dirs, rec, workers=4, log=None, limit=None):
    """PREP (CPU): the training inputs in <out>/cache and work/plan.json (internal prep_xclean.prepare, same maths)."""
    t0 = time.time()
    row = spec["name"]
    kind = "sgm32"
    for d in (dirs["work"], dirs["hot"]):
        U.ensure_dir(d)
    log = log or U.Log(os.path.join(dirs["work"], "prep.log"), prefix="[prep %s] " % row)
    fam = CFG.FAMILY[spec["family"]]
    channels = fam["channels"]
    sky_on = bool(fam.get("sky", True))
    run = RUNS.open_run(spec["run_dir"])
    ds = SRC.open_dataset(spec["sequence"], spec.get("ext"), gate=spec.get("gate", True))
    rig = SRC.rig_of(ds)
    P = depth_paths(dirs)
    didx = U.read_json(P["index"])
    if not didx or not os.path.isfile(P["arr"]):
        raise SystemExit("%s: the depth stage has not run (%s)" % (row, P["index"]))
    if U.sha1_file(P["arr"]) != didx.get("array_sha1"):
        raise SystemExit("%s changed since the depth stage: re-run it" % P["arr"])
    frames = didx["frames"]
    if frames != select_views(run, limit):
        raise SystemExit("the depth stage's frames differ from this PREP's view list (limit %s): re-run depth" % limit)
    W_full, H_full = rig["W"], rig["H"]
    K_full = rig["K"]
    sc = float(spec.get("train_scale") or 1.0)
    W, H = (int(round(W_full * sc)), int(round(H_full * sc))) if sc != 1.0 else (W_full, H_full)
    K = scale_K(K_full, sc)
    fB_t = rig["fB"] * sc
    n = len(frames)
    kf_ids = np.array(frames, int)                    # kf_id := the frame index
    times = np.array([run.t[f] for f in frames])
    Twc = [run.T[f] for f in frames]
    up, up_rule = run.up(ds)
    # the poses used, as TUM files in the work dir (the reused usability instruments read them)
    tdir = U.ensure_dir(dirs["work"] + "/poses_used")
    run.write_tum(tdir + "/trajectory_kf.csv", tdir + "/trajectory_frames.csv", frames)
    hold, probe, same_photo = holdout_roles(n, rec["holdout_every"], None, None, rec["probe_views"])
    log("%s: %d keyframes, stream %s, %dx%d, %d channel(s), %d held out, %d probe; up: %s; depth %s; sky %s" % (
        run.dir, n, run.stream, W, H, channels, int(hold.sum()), int(probe.sum()), up_rule, kind, "ON" if sky_on else "OFF (indoor family)"))
    paths = dict(img=os.path.join(dirs["hot"], "img.npy"), disp=os.path.join(dirs["hot"], "disp.npy"),
                 mask=os.path.join(dirs["hot"], "mask.npy"))
    for p in list(paths.values()) + [os.path.join(dirs["hot"], f) for f in ("seeds.npz", "views.json")] + \
            [os.path.join(dirs["work"], f) for f in ("plan.json", "sky_scan.json")]:
        U.remove_if_exists(p)                                     # stale-artefact rule
    tmp = {k: v.replace(".npy", ".building.npy") for k, v in paths.items()}
    for v in tmp.values():
        U.remove_if_exists(v)
    mm_img = np.lib.format.open_memmap(U.assert_writable(tmp["img"]), mode="w+", dtype=np.uint8, shape=(n, H, W, channels))
    mm_disp = np.lib.format.open_memmap(U.assert_writable(tmp["disp"]), mode="w+", dtype=np.float16, shape=(n, H, W))
    mm_mask = np.lib.format.open_memmap(U.assert_writable(tmp["mask"]), mode="w+", dtype=np.uint8, shape=(n, H, W))
    quality, img_sha, has_depth, scans, greys = [None] * n, [None] * n, np.zeros(n, bool), [None] * n, [None] * n
    deltas = [None] * n
    disp_f32_max_err = 0.0
    mparams = MK.row_mask_params(spec)
    x33, x33_path = None, None                          # GSKY-1 (x33g) is not shipped (masks.row_mask_params refuses it)
    with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(_view_job, j, ds, int(frames[j]), Twc[j], rig, P["arr"], j, channels, up, sky_on, kind, sc, K, W, H,
                          mparams, P["uniq"] if mparams.get("sgm32_sky_rule") in ("v3", "x33g") else P["delta"], x33_path)
                for j in range(n)]
        done = 0
        for fu in cf.as_completed(futs):
            j, img, disp, bits, q, sha, has, scan, grey, dl = fu.result()
            deltas[j] = dl
            scans[j] = scan
            greys[j] = grey
            mm_img[j] = img if channels == 3 else img[..., None]
            d16 = disp.astype(np.float16)
            mm_disp[j] = d16
            has_depth[j] = has and bool(didx["has"][j])
            m = disp > 0.5
            if m.any():
                disp_f32_max_err = max(disp_f32_max_err, float(np.max(np.abs(d16[m].astype(np.float32) - disp[m]) / disp[m])))
            mm_mask[j] = bits
            quality[j], img_sha[j] = q, sha
            done += 1
            if done % 100 == 0:
                log("pass 1: %d / %d views" % (done, n))
    if (mparams.get("sgm32_sky_rule_x30") or mparams.get("sgm32_sky_rule_x33")) and (channels == 3) == bool(all(greys)):
        raise SystemExit("X-30 / X-33: the family's sensor type (channels %d) disagrees with the images (all grey: %s)"
                         % (channels, bool(all(greys))))
    # pass 2: static-world (moving) mask + HOG people (OAK family), as v4
    n_moving_px = 0
    if fam["static_world_mask"] or fam["people_mask"]:
        p = MK.MASK_PARAMS
        for j in range(n):
            mv = np.zeros((H, W), bool)
            if fam["static_world_mask"] and has_depth[j]:
                nb = []
                for direction in (-1, 1):
                    k = j + direction
                    while 0 <= k < n and abs(times[k] - times[j]) < p["moving_min_dt_s"]:
                        k += direction
                    if 0 <= k < n and abs(times[k] - times[j]) <= p["moving_max_dt_s"] and has_depth[k]:
                        nb.append((np.asarray(mm_disp[k], np.float32), Twc[k]))
                if nb:
                    mv |= MK.moving_mask(np.asarray(mm_disp[j], np.float32), Twc[j], nb, K, fB_t)
            if fam["people_mask"]:
                g = np.asarray(mm_img[j])[..., 0] if channels == 1 else MK.grey_of(np.asarray(mm_img[j]))
                mv |= MK.hog_people(g, p["hog_weight_min"])
            if mv.any():
                mm_mask[j] = np.asarray(mm_mask[j]) | np.where(mv, BIT_MOVING, 0).astype(np.uint8)
                n_moving_px += int(mv.sum())
    proj_leak = [None] * n
    p2i = [None] * n                                      # X-33: P2' (X-28 interior) computed in PREP for x33g rows
    if sky_on and kind == "sgm32" and all(x is not None for x in deltas):
        from .losses import BIT_SKY as _BS
        p_ = MK.MASK_PARAMS
        for j in range(n):
            sky = (np.asarray(mm_mask[j]) & _BS) != 0
            if not sky.any():
                proj_leak[j] = 0.0
                p2i[j] = 0.0
                continue
            hit = np.zeros((H, W), bool)
            for direction in (-1, 1):
                k = j + direction
                while 0 <= k < n and abs(times[k] - times[j]) < p_["moving_min_dt_s"]:
                    k += direction
                if not (0 <= k < n and abs(times[k] - times[j]) <= p_["moving_max_dt_s"]):
                    continue
                dk = np.asarray(mm_disp[k], np.float32)
                meas = (dk > 0) & (deltas[k] > MK.DELTA_NOISE) & (dk >= fB_t / p_["sky_far_m"])
                ub, vb, _dp, ok = MK.reproject_disparity(np.where(meas, dk, 0.0), Twc[k], Twc[j], K, fB_t)
                ui, vi = np.rint(ub[ok]).astype(int), np.rint(vb[ok]).astype(int)
                hit[vi, ui] = True
            proj_leak[j] = float((hit & sky).mean())
            if x33 is not None:
                import cv2 as _cv2
                _dist = _cv2.distanceTransform(sky.astype(np.uint8), _cv2.DIST_L2, 3)
                p2i[j] = float((hit & sky & (_dist > 7)).mean())
        for j in range(n):
            if scans[j] is not None:
                scans[j]["proj_leak"] = proj_leak[j]
                if x33 is not None:
                    scans[j]["p2_interior"] = p2i[j]
    camfixed_info = None
    if fam["camera_fixed_mask"]:
        camfixed_info = _camera_fixed(mm_img, Twc, channels, dirs, apply=bool(rec.get("camera_fixed_reviewed")),
                                         mm_mask=mm_mask, log=log)
    for mm_ in (mm_img, mm_disp, mm_mask):
        mm_.flush()
        if getattr(mm_, "_mmap", None) is not None:
            mm_._mmap.close()
    del mm_, mm_img, mm_disp, mm_mask
    import gc
    gc.collect()
    for k in ("img", "disp", "mask"):
        U._replace_retry(tmp[k], paths[k])
    # frame quality, scale, budget, steps
    svm = SEED_VALID_MIN["sgm32"]
    dec, fq_stats = FQ.decide(quality, times, seed_valid_min=svm)
    train = ~hold & ~np.array([d["exclude_train"] for d in dec])
    seedable = train & ~np.array([d["exclude_seed"] for d in dec])
    n_train = int(train.sum())
    Pc = np.array([T[:3, 3] for T in Twc])
    L, Linfo = SC.local_scale(Pc, rec["scale_window_max_kf"], rec["scale_window_max_path_m"], rec["scale_window_slack"])
    cap, cap_info = SC.budget(spec, n_train, rec)
    S, spp = SC.steps(n_train, rec)
    if spec.get("steps_override"):
        S = int(spec["steps_override"])
        cap_info["steps_override"] = "--steps %d (user override)" % S
    # seeds: dense depth only (P1 exports no landmark cloud)
    mm_img = np.load(paths["img"], mmap_mode="r")
    mm_disp = np.load(paths["disp"], mmap_mode="r")
    mm_mask = np.load(paths["mask"], mmap_mode="r")
    stub = SimpleNamespace(landmarks=np.zeros((0, 3)), host=np.zeros(0, int), fB=rig["fB"], near=rig["near"])
    seeds, seed_info = build_seeds(stub, kf_ids, Twc, K, W, H, mm_img, mm_disp, mm_mask, train, seedable, rec, cap,
                                      channels, log, scale=sc)
    seed_info["source_codes"] = "0 = landmark (none on xclean rows), 1 = dense depth (%s)" % kind
    seed_info["rule"] = seed_info["rule"].replace("RAFT", "dense-depth (%s)" % kind)
    U.atomic_write_bytes(os.path.join(dirs["hot"], "seeds.npz"), _npz_bytes(**seeds))
    views = []
    for j in range(n):
        role = "hold" if hold[j] else ("train" if train[j] else "excluded")
        views.append(dict(j=j, kf_id=int(kf_ids[j]), frame_index=int(frames[j]), t=float(times[j]),
                          T_wc=np.asarray(Twc[j], float).tolist(), role=role, probe=bool(probe[j]), same_photo=False,
                          has_depth=bool(has_depth[j]), quality=quality[j], exclude=dec[j], img_sha1=img_sha[j],
                          cache_digest=didx["sha1"][j], run_state=run.state[int(frames[j])]))
    U.atomic_json(os.path.join(dirs["hot"], "views.json"), dict(row=row, views=views))
    sky_frames = [x for x in scans if x is not None]
    sky_verdict = MK.sky_scan_verdict(sky_frames, [times[j] for j in range(n) if scans[j] is not None]) if sky_frames else None
    if sky_verdict is not None and any(x is not None for x in proj_leak):
        pl = np.array([x for x in proj_leak if x is not None])
        sky_verdict["proj_leak"] = dict(max=float(pl.max()), p99=float(np.percentile(pl, 99)), n_over_1pct=int((pl > 0.01).sum()),
                                        pass_=bool((pl <= 0.01).all()),
                                        rule="X-26.6: share of the frame that is confident sky AND hit by MEASURED SGM32 points "
                                             "nearer than 20 m reprojected from the time neighbours; FAIL if any keyframe > 1 %")
    if x33 is not None and sky_verdict is not None:
        # X-33 PREP for x33g rows (registered; declared restatement): prep_pass = the scan's two MISSED clauses (unchanged
        # bars) AND P2' (X-28 interior leak, points measured under X-29, FAIL if any keyframe > 1 %); the LEAK clause of
        # the scan is reported, not a refusal (its grey proxy is a subset of sky by construction)
        pm = MK.MASK_PARAMS
        missed_ok = bool(sky_verdict["missed_valid_frame_share"] <= pm["scan_miss_frames_max"]
                         and sky_verdict["missed_valid_median"] <= pm["scan_miss_median_max"])
        a2 = np.array([x for x in p2i if x is not None]) if all(x is not None for x in p2i) else None
        p2 = dict(max=float(a2.max()) if a2 is not None else None,
                  p99=float(np.percentile(a2, 99)) if a2 is not None else None,
                  n_over_1pct=int((a2 > 0.01).sum()) if a2 is not None else None,
                  pass_=bool(a2 is not None and (a2 <= 0.01).all()),
                  worst=([dict(kf_id=int(kf_ids[i]), t=round(float(times[i]), 2), share=round(float(a2[i]), 5))
                          for i in np.argsort(-a2)[:8]] if a2 is not None else None),
                  rule="X-28 P2': share of the frame that is confident-sky INTERIOR (> 7 px from non-sky) AND hit by SGM32 "
                       "points measured under X-29 (u > 13.93) nearer than 20 m, reprojected from the time neighbours; "
                       "FAIL if any keyframe > 1 % (fails closed when u is missing)")
        sx = [s for s in scans if s is not None]
        sky_verdict["x33g"] = dict(
            missed_pass=missed_ok, p2_interior=p2, prep_pass=bool(missed_ok and p2["pass_"]),
            leak_clause_reported=dict(n_leak_frames=sky_verdict["n_leak_frames"], leak_max=sky_verdict["leak_max"],
                                      would_pass=bool(sky_verdict["n_leak_frames"] == 0)),
            model_sky_share_p10_50_90=[float(np.percentile([s.get("x33_model_sky", 0.0) for s in sx], q)) for q in (10, 50, 90)],
            veto_share_p10_50_90=[float(np.percentile([s.get("x33_veto", 0.0) for s in sx], q)) for q in (10, 50, 90)],
            veto_share_max=float(max(s.get("x33_veto", 0.0) for s in sx)),
            model_sha1=x33["model_sha1"], skyprob_array_sha1=x33["array_sha1"],
            rule="X-33 (registered 2026-10-27 00:02): prep_pass = missed clauses AND P2'; leak clause reported only")
    U.atomic_json(os.path.join(dirs["work"], "sky_scan.json"), dict(row=row, verdict=sky_verdict, sky_on=sky_on,
                                                                    frames=[dict(j=j, t=float(times[j]), **(scans[j] or {})) for j in range(n)]))
    pose_file = run.pose_file
    inputs = dict(kit_traj_sha1=U.sha1_file(tdir + "/trajectory_kf.csv"), outer_traj_sha1=pose_file["sha1"],
                  pose_file=pose_file["path"], run_files={k: v["sha1"] for k, v in run.files.items()},
                  calib_sha1=U.sha1_file(os.path.join(ds.path, "calib.json")), rect_digest=rig["rect_digest"],
                  depth_index_sha1=U.sha1_file(P["index"]), depth_array_sha1=didx["array_sha1"],
                  cache_header_digest=U.sha1_json(didx["sha1"]), image_bytes_digest=U.sha1_json(img_sha),
                  depth_key=(didx.get("contract") or {}).get("key"),
                  views_sha1=U.sha1_file(os.path.join(dirs["hot"], "views.json")),
                  seeds_sha1=U.sha1_file(os.path.join(dirs["hot"], "seeds.npz")),
                  img_npy_sha1=U.sha1_file(paths["img"]), disp_npy_sha1=U.sha1_file(paths["disp"]),
                  mask_npy_sha1=U.sha1_file(paths["mask"]),
                  )
    plan = dict(
        row=row, prep_version=PREP_VERSION, source="xclean_%s" % run.layout, run_dir=run.dir, draw=os.path.basename(run.dir),
        draw_dir=run.dir, kit=None, pose_stream=run.stream, run_checks=run.no_gt_checks(), engine=run.engine_identity(),
        family=spec["family"], preset=spec.get("preset"),
        dataset=ds.path.replace("\\", "/"), integrity_blockers=ds.blockers,
        depth_source=kind, depth_contract=didx.get("contract"), depth_licence_clean=didx.get("licence_clean"),
        depth_valid_share=didx.get("valid_share_stats"), xclean_modules=didx.get("xclean_modules"),
        n_kf_total=len(run.kf_frames or []), n_frames_published=len(run.frames), n_views=n, limit=limit, image_size=[W, H],
        channels=channels, grey_data=bool(all(greys)), K_rect=K.tolist(), fB=fB_t, near_clip_m=rig["near"], max_depth_m=rig["max_depth"],
        cache_dir=None, train_scale=sc, image_size_full=[W_full, H_full], K_rect_full=K_full.tolist(), fB_full=rig["fB"],
        camera_rules=dict(fB=rig["fB_rule"], near=rig["near_rule"], far=rig["far_rule"],
                          K="P1[:3,:3] of cv2.stereoRectify(alpha=0) (xclean io/dataset.py rectify_maps)"),
        pixel_convention="OpenCV: pixel (i, j) centred at image coordinates (j, i); renderers add 0.5 for gsplat (review F5)",
        sky_on=sky_on, sky_scan=sky_verdict,
        prep_pass=bool(sky_verdict is None or ((sky_verdict.get("x33g") or {}).get("prep_pass") if x33 is not None
                                               else sky_verdict["pass"])),
        depth_model="z = fB / d, no disparity offset; d = 0 is INVALID (no measurement)" if kind == "sgm32" else
                    "z = fB / d (RAFT cache; A/B measurement only)",
        up=up.tolist(), up_rule=up_rule,
        n_hold=int(hold.sum()), n_probe=int(probe.sum()), n_same_photo=0, n_train=n_train,
        n_excluded_train=int((~hold & ~train).sum()), n_seedable=int(seedable.sum()),
        n_with_depth=int(has_depth.sum()), n_missing_depth=int((~has_depth).sum()),
        missing_depth_kf=[int(kf_ids[j]) for j in np.nonzero(~has_depth)[0]],
        excluded=[dict(kf_id=int(kf_ids[j]), t=float(times[j]), role="hold" if hold[j] else "train", reasons=dec[j]["reasons"])
                  for j in range(n) if dec[j]["exclude_train"]],
        frame_quality_stats=dict(fq_stats, seed_valid_min=svm, **{k: U.stats([q[k] for q in quality if q[k] is not None])
                                                                  for k in ("dark_share", "sat_share", "mean_intensity", "valid_share", "lap_var")}),
        L=L, L_info=Linfo, cap_max=cap, cap_info=cap_info, steps=S, steps_per_photo=spp,
        model_gib=SC.model_gib(cap, rec["bytes_per_gaussian"]),
        inputs_gb=float(n * H * W * (channels + 2 + 1) / 1e9),
        disp_fp16_max_rel_err=disp_f32_max_err, moving_px_total=n_moving_px, camera_fixed=camfixed_info,
        privacy=MK.privacy_blur(None)[1], seeds=seed_info, inputs=inputs, recipe=rec, label=spec.get("label"),
        mask_params=mparams, mask_overrides=spec.get("mask_overrides"), fq_rules=FQ.FQ, prep_seconds=round(time.time() - t0, 1))
    plan["config_hash"] = CFG.config_hash(dict(recipe=CFG.prep_recipe(rec), mask=mparams, fq=FQ.FQ, prep=PREP_VERSION,
                                               limit=limit, train_scale=sc, inputs=inputs, depth=kind, svm=svm,
                                               steps_override=spec.get("steps_override")))
    U.atomic_json(os.path.join(dirs["work"], "plan.json"), plan)
    log("done in %.0f s: n_train %d, hold %d, excluded %d, L %.1f m, cap %d, steps %d, seeds %d (%s)"
        % (time.time() - t0, n_train, int(hold.sum()), plan["n_excluded_train"], L, cap, S, seed_info["n_total"], kind))
    return plan


class Prepared:
    """Read-only view of a prepared map (memmaps) for the trainer / eval / export (internal prep_row.Prepared)."""

    def __init__(self, dirs, name=None):
        self.dirs = dirs
        self.plan = U.read_json(os.path.join(dirs["work"], "plan.json"))
        if self.plan is None:
            raise SystemExit("not prepared (no %s/plan.json): run the prep stage" % dirs["work"])
        self.row = name or self.plan.get("row")
        vj = U.read_json(os.path.join(dirs["hot"], "views.json"))
        self.views = vj["views"]
        self.array_files = {k: os.path.join(dirs["hot"], k + ".npy") for k in ("img", "disp", "mask")}
        self.img = np.load(os.path.join(dirs["hot"], "img.npy"), mmap_mode="r")
        self.disp = np.load(os.path.join(dirs["hot"], "disp.npy"), mmap_mode="r")
        self.mask = np.load(os.path.join(dirs["hot"], "mask.npy"), mmap_mode="r")
        z = np.load(os.path.join(dirs["hot"], "seeds.npz"))
        self.seeds = {k: z[k] for k in z.files}
        self.K = np.array(self.plan["K_rect"], float)
        self.W, self.H = self.plan["image_size"]
        self.channels = self.plan["channels"]
        self.fB = float(self.plan["fB"])
        self.up = np.array(self.plan["up"], float)
        self.train_ids = [v["j"] for v in self.views if v["role"] == "train"]
        self.hold_ids = [v["j"] for v in self.views if v["role"] == "hold"]
        self.probe_ids = [v["j"] for v in self.views if v["probe"]]

    def T_wc(self, j):
        return np.array(self.views[j]["T_wc"], float)
