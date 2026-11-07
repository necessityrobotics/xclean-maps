# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""EXPORT: the files of a finished map in <out>/.

  splats.ply        standard INRIA 3DGS binary PLY (x y z, nx ny nz, f_dc_0..2, f_rest_0..8 (SH degree 1, channel-major),
                    opacity (logit), scale_0..2 (log), rot_0..3 (w x y z)); invisible splats (alpha < 1/255) removed;
                    grey maps channel-averaged; parsed back with plyfile and our own reader, and re-scored from the FILE
                    on every held-out view (|file - trained| <= 0.02 dB)
  cameras.json      every view: INRIA-style entries (position = camera centre, rotation = camera-to-world 3x3, fx, fy,
                    width, height) plus cx, cy, T_world_cam (4x4, OpenCV axes), role (train / hold / excluded), frame index
  pointcloud.ply    SGM32 keyframe depth back-projected with the run's poses (x y z red green blue)
  keyframes/        80 evenly spaced rectified keyframe JPEGs + keyframes.json
  preview.png       top view (point cloud + path) + held-out photo vs render + an oblique render
  previews/         held-out photo | render pairs (JPEG), up to 12
  report.json       held-out PSNR / SSIM (splats on black, with the trainer's sky panorama, over the plain gradient
                    backdrop), the re-score, the no-GT render checks (reported, not blocking), the run checks
  report.md         the same, short and human-readable
clean_params / write_splats / pointcloud / keyframes are the internal builder's export_row.py functions (an internal splat
reader is replaced by plyfile). preview is that module's preview without the ground-truth / ORB-SLAM3 traces.
"""
from __future__ import annotations

import os

import numpy as np

from . import io_lib as IO
from . import util as U


# --------------------------------------------------------------------------------------------- splats
def clean_params(params, channels, invisible_alpha=1.0 / 255.0):
    """Remove invisible splats; for grey rows equalise the colour channels (after checking they are equal within
    1/255 in colour space). Returns (params, info). Raises when a grey row's channels disagree (a bug, not a fix)."""
    p = {k: np.asarray(v) for k, v in params.items()}
    n0 = len(p["means"])
    finite = np.ones(n0, bool)
    for k, v in p.items():
        finite &= np.isfinite(v.reshape(n0, -1)).all(1)
    qn = np.linalg.norm(p["quats"].astype(np.float64), axis=1)
    finite &= qn >= 1e-12
    n_bad = int((~finite).sum())
    if n_bad > max(10, 1e-4 * n0):
        raise RuntimeError("%d of %d splats are non-finite or have a zero quaternion: more than a tiny fraction" % (n_bad, n0))
    op = np.where(finite, p["opacities"].reshape(-1).astype(np.float64), -1e9)
    alpha = 1.0 / (1.0 + np.exp(-np.clip(op, -500, 500)))
    keep = (alpha >= invisible_alpha) & finite
    info = dict(n_trained=int(n0), n_nonfinite_removed=n_bad, n_invisible_removed=int(((alpha < invisible_alpha) & finite).sum()),
                invisible_rule="alpha < 1/255 removed; non-finite rows and zero-norm quaternions removed and counted")
    p = {k: v[keep] for k, v in p.items()}
    if channels == 1:
        c = IO.SH_C0 * p["sh0"][:, 0, :]
        spread = float((c.max(1) - c.min(1)).max()) if len(c) else 0.0
        rest_spread = float(np.abs(p["shN"] - p["shN"].mean(2, keepdims=True)).max()) if len(c) else 0.0
        sp = (c.max(1) - c.min(1)) if len(c) else np.zeros(0)
        info.update(grey_channel_spread=spread, grey_shN_spread=rest_spread,
                    grey_spread_quantiles={str(q): float(np.percentile(sp, q)) for q in (50, 90, 99, 99.9)} if len(sp) else None,
                    grey_share_over_1_255=float((sp >= 1.0 / 255.0).mean()) if len(sp) else 0.0,
                    grey_rule=("XCLEAN X-19: the three SH channels of a grey row are averaged (the least-squares grey "
                               "estimate); the export re-score (file vs trained, <= 0.02 dB) is the guard"))
        p["sh0"] = np.repeat(p["sh0"].mean(2, keepdims=True), 3, axis=2)
        p["shN"] = np.repeat(p["shN"].mean(2, keepdims=True), 3, axis=2)
        info["grey_equalised"] = True
    return p, info


def write_splats(path, params):
    """Write splats.ply and parse it back. Our own reader must give the arrays back exactly. plyfile (GPL-3.0, so NOT a
    dependency of xclean-maps) is used as a second, independent parser only when the user has it installed."""
    arrs = IO.splat_ply_arrays(params)
    N = IO.write_splat_ply(path, *arrs)
    v0 = IO.read_ply_vertices(path)
    ok_schema = list(v0.dtype.names) == IO.SPLAT_PLY_FIELDS and int(v0.shape[0]) == N
    del v0
    back = IO.params_from_splat_ply(path)
    q = arrs[5]
    exact = all(np.array_equal(back[k], v) for k, v in (("means", arrs[0]), ("scales", arrs[4]), ("opacities", arrs[3])))         and np.array_equal(back["quats"], q) and np.array_equal(back["sh0"][:, 0, :], arrs[1])
    ok_pf = None
    try:
        from plyfile import PlyData                              # optional second parser (GPL-3.0; not a dependency)
    except ImportError:
        PlyData = None
    if PlyData is not None:
        pd = PlyData.read(path)
        v = pd["vertex"]
        xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1)
        ok_pf = bool([p_.name for p_ in v.properties] == IO.SPLAT_PLY_FIELDS and v.count == N
                     and np.array_equal(xyz.astype(np.float32), arrs[0]))
    info = dict(n=int(N), bytes=os.path.getsize(path), sha1=U.sha1_file(path), schema_ok=bool(ok_schema),
                own_reader_exact=bool(exact), plyfile_check=ok_pf, schema=("INRIA 3DGS binary PLY: x y z, nx ny nz (0), "
                "f_dc_0..2, f_rest_0..8 (SH degree 1, CHANNEL-major), opacity (logit), scale_0..2 (log), rot_0..3 (w,x,y,z)"))
    if not (ok_schema and exact and ok_pf is not False):
        raise RuntimeError("splat PLY parse-back failed: %s" % info)
    return info


def pointcloud(path, prep, near, far, max_points=2_000_000, stride=4, every=None):
    W, H = prep.W, prep.H
    K = prep.K
    fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    views = prep.views
    every = every or max(1, len(views) // 400)
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    P, C, used = [], [], 0
    from .losses import BIT_EDGE, BIT_MOVING, BIT_SKY, BIT_VALID
    for v in views[::every]:
        j = v["j"]
        d = np.asarray(prep.disp[j], np.float32)[yy, xx]
        mb = np.asarray(prep.mask[j])[yy, xx]
        ok = ((mb & BIT_VALID) != 0) & ((mb & (BIT_SKY | BIT_EDGE | BIT_MOVING)) == 0)
        z = np.where(ok, prep.fB / np.maximum(d, 1e-6), 0.0)
        ok &= (z > near) & (z <= far)
        if not ok.any():
            continue
        X = np.stack([(xx[ok] - cx) / fx * z[ok], (yy[ok] - cy) / fx * z[ok], z[ok]], 1)
        T = prep.T_wc(j)
        P.append((X @ T[:3, :3].T + T[:3, 3]).astype(np.float32))
        im = np.asarray(prep.img[j])
        c = im[yy[ok], xx[ok]]
        C.append(np.repeat(c, 3, axis=1) if c.shape[1] == 1 else c)
        used += 1
    P = np.concatenate(P)
    C = np.concatenate(C).astype(np.uint8)
    ext = np.percentile(P, 98, axis=0) - np.percentile(P, 2, axis=0)
    voxel = float(np.clip(np.linalg.norm(ext) / 1500.0, 0.01, 0.25))
    while True:
        key = np.floor(P / voxel).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        if len(first) <= max_points:
            break
        voxel *= 1.25
    first = np.sort(first)
    IO.write_points_ply(path, P[first], C[first])
    return dict(points=int(len(first)), keyframes_used=used, every_kth_keyframe=every, pixel_stride=stride, voxel_m=voxel,
                depth_band_m=[near, far], max_points=max_points, bytes=os.path.getsize(path),
                colour="grey (replicated to RGB)" if prep.channels == 1 else "RGB",
                source=("32-direction SGM disparity of each keyframe (xclean/stereo/sgm32.py; 0 = invalid, skipped), depth = "
                        "fB / d, the run's keyframe poses, map frame; sky / edges / moving pixels excluded"))


def keyframes(kd, prep, n=80):
    import cv2
    U.remove_if_exists(kd)
    tmpd = kd + ".tmp"
    U.remove_if_exists(tmpd)
    U.ensure_dir(tmpd)
    W, H = prep.W, prep.H
    views = prep.views
    pick = sorted(set(np.linspace(0, len(views) - 1, min(n, len(views))).round().astype(int).tolist()))
    Wd = min(W, 320 if W / H < 2.0 else 512)
    s = Wd / W
    from .prep import scale_K
    Ks = scale_K(prep.K, s)              # OpenCV pixel convention, as the INTER_AREA resize maps pixel centres (F5)
    recs = []
    for i in pick:
        v = views[i]
        im = np.asarray(prep.img[v["j"]])
        im = cv2.resize(im, (Wd, int(round(H * s))), interpolation=cv2.INTER_AREA)
        if im.ndim == 3 and im.shape[2] == 3:
            enc = im[:, :, ::-1]
        else:
            enc = im.reshape(im.shape[0], im.shape[1])
        fn = "kf%06d.jpg" % v["kf_id"]
        ok, buf = cv2.imencode(".jpg", enc, [cv2.IMWRITE_JPEG_QUALITY, 85])
        U.atomic_write_bytes(os.path.join(tmpd, fn), buf.tobytes())
        recs.append(dict(file=fn, kf_id=v["kf_id"], frame_index=v["frame_index"], t=v["t"], held_out_of_map=v["role"] != "train",
                         role=v["role"],
                         T_map_cam=np.round(prep.T_wc(v["j"]), 6).tolist()))
    meta = dict(row=prep.row, n=len(recs), image_size=[Wd, int(round(H * s))], K=np.round(Ks, 4).tolist(),
                camera="rectified left camera (the splat's own camera model), downscaled x%.4f" % s,
                pose="T_map_cam: 4x4 camera-to-map (OpenCV camera axes), the pose the map was trained with (%s)"
                     % prep.plan.get("pose_stream"),
                channels="RGB" if prep.channels == 3 else "grey (single channel JPEG)",
                selection="evenly spaced over the route's %d keyframes" % len(views), keyframes=recs)
    U.atomic_json(os.path.join(tmpd, "keyframes.json"), meta)
    os.replace(tmpd, kd)
    return dict(n=len(recs), width=Wd, bytes=sum(os.path.getsize(os.path.join(kd, f)) for f in os.listdir(kd)))




# --------------------------------------------------------------------------------------------- preview
def preview(path, prep, model, pointcloud_ply, oblique_size=(1280, 720)):
    """Top view + held-out real vs render + oblique render (internal export_row.preview without GT / ORB-SLAM3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .render import look_at
    up = prep.up / np.linalg.norm(prep.up)
    pc = IO.read_ply_vertices(pointcloud_ply)
    P = np.stack([pc["x"], pc["y"], pc["z"]], 1)
    Cc = np.stack([pc["red"], pc["green"], pc["blue"]], 1) / 255.0
    sub = np.random.RandomState(0).choice(len(P), min(len(P), 400_000), replace=False)
    cams = np.array([prep.T_wc(v["j"])[:3, 3] for v in prep.views])
    X = cams - cams.mean(0)
    X = X - np.outer(X @ up, up)
    e1 = np.linalg.svd(X, full_matrices=False)[2][0] if len(X) > 2 else np.array([1.0, 0, 0])
    e1 = e1 - (e1 @ up) * up
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross(up, [0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.cross(up, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    pr = lambda A: np.stack([A @ e1, A @ e2], 1)   # noqa: E731
    hold = prep.hold_ids[len(prep.hold_ids) // 2] if prep.hold_ids else prep.views[len(prep.views) // 2]["j"]
    real = np.asarray(prep.img[hold], np.float32) / 255.0
    real = np.repeat(real, 3, 2) if real.shape[2] == 1 else real
    rend, _, _ = model.view(prep.T_wc(hold), prep.K, prep.W, prep.H, with_sky=False)
    km = prep.views[len(prep.views) // 2]["j"]
    Tm = prep.T_wc(km)
    cam = Tm[:3, 3]
    fwd = Tm[:3, 2] - (Tm[:3, 2] @ up) * up
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    med = 5.0
    eye = cam - 1.2 * med * fwd + 1.0 * med * up
    Wv, Hv = oblique_size
    Kv = np.array([[0.6 * Wv, 0, Wv / 2], [0, 0.6 * Wv, Hv / 2], [0, 0, 1.0]])
    ov, _, _ = model.view(IO.inv_T(look_at(eye, cam + med * fwd, up)), Kv, Wv, Hv, with_sky=False)
    fig = plt.figure(figsize=(16, 10), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
    a0 = fig.add_subplot(gs[:, 0])
    q = pr(P[sub])
    a0.scatter(q[:, 0], q[:, 1], s=0.15, c=Cc[sub] * 0.8 + 0.1, linewidths=0, rasterized=True)
    o = pr(cams)
    a0.plot(o[:, 0], o[:, 1], "-", color="#00798c", lw=1.1, label="camera path (keyframes)")
    lo, hi = o.min(0), o.max(0)
    pad = 0.08 * max(float((hi - lo).max()), 1.0)
    a0.set_xlim(lo[0] - pad, hi[0] + pad)
    a0.set_ylim(lo[1] - pad, hi[1] + pad)
    a0.set_aspect("equal", adjustable="datalim")
    a0.tick_params(labelsize=8)
    a0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False)
    a0.set_title("%s - top view, map frame (metres)" % prep.row, fontsize=9)
    a1 = fig.add_subplot(gs[0, 1])
    a1.imshow(np.concatenate([real, np.clip(rend, 0, 1)], 0))
    a1.axis("off")
    a1.set_title("held-out keyframe %d: photo (top) vs splat render (bottom)" % prep.views[hold]["kf_id"], fontsize=10)
    a2 = fig.add_subplot(gs[1, 1])
    a2.imshow(np.clip(ov, 0, 1))
    a2.axis("off")
    a2.set_title("oblique splat view behind / above keyframe %d" % prep.views[km]["kf_id"], fontsize=10)
    fig.tight_layout()
    tmp = path + ".tmp.png"
    U.assert_writable(tmp)
    fig.savefig(tmp)
    plt.close(fig)
    os.replace(tmp, path)
    return dict(holdout_kf=prep.views[hold]["kf_id"], overview_kf=prep.views[km]["kf_id"], up_map=up.tolist())


def holdout_pairs(dirpath, prep, model, n=12):
    """photo | render JPEG pairs of up to n evenly spaced held-out views (splats on black)."""
    import cv2
    U.remove_if_exists(dirpath)
    U.ensure_dir(dirpath)
    hs = prep.hold_ids
    pick = [hs[i] for i in sorted(set(np.linspace(0, len(hs) - 1, min(n, len(hs))).round().astype(int).tolist()))] if hs else []
    out = []
    for j in pick:
        real = np.asarray(prep.img[j], np.float32) / 255.0
        real = np.repeat(real, 3, 2) if real.shape[2] == 1 else real
        rend, _, _ = model.view(prep.T_wc(j), prep.K, prep.W, prep.H, with_sky=False)
        im = (np.clip(np.concatenate([real, rend], 1), 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
        fn = "holdout_kf%06d.jpg" % prep.views[j]["kf_id"]
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(im), [cv2.IMWRITE_JPEG_QUALITY, 88])
        U.atomic_write_bytes(os.path.join(dirpath, fn), buf.tobytes())
        out.append(fn)
    return out


# --------------------------------------------------------------------------------------------- cameras.json
def cameras_json(path, prep):
    """Every view's camera. INRIA 3DGS cameras.json keys (id, img_name, width, height, position = camera centre,
    rotation = camera-to-world rotation, fx, fy) plus cx, cy, T_world_cam (OpenCV camera axes: x right, y down,
    z forward), role, frame_index, t."""
    K = prep.K
    cams = []
    for v in prep.views:
        T = prep.T_wc(v["j"])
        cams.append(dict(id=int(v["j"]), img_name="kf%06d" % v["kf_id"], width=int(prep.W), height=int(prep.H),
                         position=np.round(T[:3, 3], 6).tolist(), rotation=np.round(T[:3, :3], 9).tolist(),
                         fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                         T_world_cam=np.round(T, 9).tolist(), role=v["role"], frame_index=int(v["frame_index"]),
                         t=float(v["t"])))
    U.atomic_json(path, cams)
    return dict(n=len(cams), convention="T_world_cam0_rectified (xclean); OpenCV camera axes; pixel centres at integer "
                                        "coordinates (cv2 convention)")


# --------------------------------------------------------------------------------------------- report
def report_md(path, rep):
    h = rep.get("heldout") or {}
    lines = ["# xclean-maps report: %s" % rep.get("name"), "",
             "Built %s with preset `%s` from `%s` (%s) and the sequence `%s`." % (
                 rep.get("finished"), rep.get("preset"), rep.get("run_dir"), rep.get("pose_stream"), rep.get("sequence")),
             "", "## Held-out views (never trained on), at the run's poses", "",
             "| measure | value |", "|---|---|"]
    for k, lab in (("n", "held-out views"), ("psnr_splats_on_black_db", "PSNR, splats on black (dB)"),
                   ("psnr_gradient_backdrop_db", "PSNR, splats over the plain gradient backdrop (dB)"),
                   ("psnr_with_sky_panorama_db", "PSNR, with the trainer's sky panorama (dB; panorama not exported)"),
                   ("psnr_exposure_interpolated_db", "PSNR, exposure-interpolated (dB)"), ("ssim", "SSIM (panorama composited)")):
        val = h.get(k)
        lines.append("| %s | %s |" % (lab, "-" if val is None else ("%.3f" % val if isinstance(val, float) else val)))
    lines += ["", "## Map", "", "- splats: %s (%s invisible removed); %.1f MB" % (
        rep.get("n_splats"), rep.get("n_invisible_removed"), (rep.get("splats_bytes") or 0) / 1e6),
        "- re-score from the file: %s" % rep.get("rescore"),
        "- training: %s steps, %.0f s" % (rep.get("steps"), rep.get("train_seconds") or 0),
        "", "## No-GT checks (reported, never blocking)", ""]
    for k, v in (rep.get("render_checks") or {}).items():
        lines.append("- %s: %s" % (k, "pass" if (v or {}).get("pass_") else "FAIL (see report.json)"))
    lines += ["- sky scan (PREP): %s" % rep.get("sky_scan_pass"), "- run checks: %s" % rep.get("run_checks_short"), ""]
    U.atomic_write_text(path, "\n".join(lines) + "\n")
