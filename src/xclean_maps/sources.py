# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Images, camera model and dense depth of a map, all through the INSTALLED xclean package (no other xclean copy).

  open_dataset(folder)  xclean.io.dataset.Dataset (xclean's own reader, rectification and integrity gate)
  rig_of(ds)            the RECTIFIED left camera: K = P1[:3, :3] of cv2.stereoRectify(alpha=0), fB = -P2[0, 3]
                        (xclean core/derive.py Rig.fB), near clip / max depth = xclean's Rig derivations
  load_left_colour      the left image in its own channels, remapped with the engine's own map
  Sgm32Source           our 32-direction SGM (xclean/stereo/sgm32.py SgmDepth) on the grey rectified pair exactly as
                        the engine reads it (Dataset.images). GPU (CuPy). Contract: float32 disparity in px,
                        0 = INVALID (no measurement, never "far").
  accel_up              up from the accelerometer mean over the first second (when the engine gave no gravity)

These functions are the internal builder's sources.py, unchanged in their maths. Removed: the frozen-engine import root,
the row registry, the licence fence and the RAFT A/B source (see CHANGES_FROM_INTERNAL.md).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

XCLEAN_MODULES = ("stereo/sgm32.py", "io/dataset.py", "core/derive.py", "stages/cache.py")


def detect_ext(folder):
    """'png' unless the first frame of timestamps.csv exists only as .jpg (xclean.io.rows._detect_ext's rule)."""
    try:
        with open(os.path.join(folder, "timestamps.csv"), encoding="utf-8") as f:
            f.readline()
            for ln in f:
                p = ln.strip().split(",")
                if len(p) >= 2:
                    i = int(p[0])
                    break
            else:
                return "png"
    except (OSError, ValueError):
        return "png"
    png = os.path.isfile(os.path.join(folder, "left", "%06d.png" % i))
    jpg = os.path.isfile(os.path.join(folder, "left", "%06d.jpg" % i))
    return "jpg" if (jpg and not png) else "png"


def open_dataset(folder, ext=None, gate=True):
    """The xclean Dataset (integrity gate ON by default: the same reader and rectification the engine uses)."""
    from xclean.io.dataset import Dataset
    if not os.path.isfile(os.path.join(folder, "calib.json")):
        raise SystemExit("%s is not an xclean sequence folder (no calib.json; see xclean docs/SEQUENCE_FORMAT.md)" % folder)
    return Dataset(folder, ext=ext or detect_ext(folder), gate=gate)


def xclean_module_hashes():
    import xclean
    root = os.path.dirname(os.path.abspath(xclean.__file__))
    out = {}
    for rel in XCLEAN_MODULES:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            h = hashlib.sha1()
            with open(p, "rb") as f:
                h.update(f.read())
            out[rel] = h.hexdigest()
    return dict(xclean_version=getattr(xclean, "__version__", None), root=root.replace("\\", "/"), modules=out)


def rig_of(ds):
    """Camera model of the RECTIFIED left camera (the splat's camera) from the dataset's own rectification:
    K = P1[:3, :3] of cv2.stereoRectify(alpha=0) (io/dataset.py rectify_maps), fB = -P2[0, 3] (core/derive.py Rig.fB),
    near clip / max depth = the Class-B derivations of core/derive.py."""
    (m1x, m1y), (m2x, m2y), P1, P2, Q, R1, R2 = ds.rectify_maps()
    K = np.array(P1, float)[:3, :3].copy()
    if abs(K[0, 1]) > 1e-9 or abs(K[0, 0] - K[1, 1]) > 1e-6 * K[0, 0]:
        raise SystemExit("rectified P1 is not [f 0 cx; 0 f cy]: %s" % K.tolist())
    W, H = int(ds.calib["camera_width"]), int(ds.calib["camera_height"])
    return dict(K=K, W=W, H=H, fB=float(ds.rig.fB), near=float(ds.rig.near_clip_m), max_depth=float(ds.rig.max_depth_m),
                m1=(m1x, m1y), m2=(m2x, m2y), rect_digest=ds.rect_digest(), fB_rule="-P2[0,3] of cv2.stereoRectify "
                "(xclean core/derive.py Rig.fB)", near_rule="Rig.near_clip_m = 1.10 x fB / 128", far_rule="Rig.max_depth_m = 0.2 x fB")


def load_left_colour(ds, index, m1):
    """Rectified LEFT image for the map: the file's own channels (colour rows keep colour; RGB order), remapped with the
    SAME map the engine's grey reader uses (Dataset.rectify_maps m1, INTER_LINEAR). -> (img uint8 HxW or HxWx3, raw
    bytes sha1)."""
    import cv2
    f = ds.frames()[index]
    with open(f.left, "rb") as fh:
        raw = fh.read()
    img0 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img0 is None:
        raise IOError("unreadable %s" % f.left)
    if img0.ndim == 3 and img0.shape[2] == 4:
        img0 = img0[:, :, :3]
    grey = img0.ndim == 2 or (np.array_equal(img0[..., 0], img0[..., 1]) and np.array_equal(img0[..., 1], img0[..., 2]))
    if grey:
        g = img0 if img0.ndim == 2 else cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        out = cv2.remap(g, m1[0], m1[1], cv2.INTER_LINEAR)
    else:
        out = cv2.remap(img0, m1[0], m1[1], cv2.INTER_LINEAR)[:, :, ::-1]
    return np.ascontiguousarray(out), hashlib.sha1(raw).hexdigest(), bool(grey)


def uniq_margin(C, xp=np):
    """X-29 u(p) from a raw census cost volume C (H, W, D) uint8, with xp = numpy or cupy (same maths): d* = the first
    argmin over the in-range d <= x; u = min over in-range |d - d*| >= 2 of C - C(d*); undefined (no such d, or x < 3)
    -> 0. -> uint8 (H, W)."""
    C = C.astype(xp.int16)
    H, W, D = C.shape
    dd = xp.arange(D, dtype=xp.int16)[None, None, :]
    xs = xp.arange(W, dtype=xp.int16)[None, :, None]
    Cm = xp.where(dd <= xs, C, xp.int16(255))
    ds = xp.argmin(Cm, axis=2)
    mn = xp.take_along_axis(Cm, ds[..., None], axis=2)[..., 0]
    adj = xp.abs(dd - ds[..., None].astype(xp.int16)) < 2
    sec = xp.where(adj, xp.int16(255), Cm).min(axis=2)
    u = xp.where(sec >= 255, 0, sec - mn)
    u[:, :3] = 0
    return xp.clip(u, 0, 80).astype(xp.uint8)


# ------------------------------------------------------------------------------------------------ depth
class Sgm32Source:
    """Our dense depth: xclean SgmDepth on the grey rectified pair (Dataset.images, exactly the engine's input)."""
    kind = "sgm32"
    licence_clean = True

    def __init__(self, verbose=False):
        from xclean.stereo.sgm32 import SgmDepth, MAX_DISP, MIN_DISP_PX, LR_THRESHOLD_PX, UNIQUENESS, P1, P2, _kernel_sha1
        self.stage = SgmDepth(verbose=verbose)
        self.contract = dict(algorithm="32-direction SGM (xclean/stereo/sgm32.py SgmDepth), census 9x9 (80 bits), "
                                       "one aggregation, left-right check, no hole fill",
                             invalid="0 = INVALID (no measurement, never 'far')", max_disp=MAX_DISP, min_disp_px=MIN_DISP_PX,
                             lr_threshold_px=LR_THRESHOLD_PX, uniqueness=UNIQUENESS, p1=P1, p2=P2,
                             kernel_sha1=_kernel_sha1(), key=self.stage.key())
        self.n = 0

    def disparity(self, ds, index):
        l, r = ds.images(index)                       # rectified grey pair, the engine's own reader
        d = self.stage.compute(l, r).astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        self.n += 1
        return d

    def uniqueness_margin(self):
        """X-29 (registered 19:25): u(p) = min over in-range d with |d - d*| >= 2 of C(p, d) - C(p, d*), d* = the first
        argmin of SGM32's RAW census cost over the in-range disparities d <= x (out-of-range costs excluded). No
        non-adjacent in-range cost (x < 3, or 3 in-range values with d* in the middle): u undefined -> 0 (unmeasured).
        Ties with another exact minimum at |d - d*| >= 2 -> 0. -> uint8 (H, W). Computed with CuPy on a re-upload of
        the public cost_volume()."""
        import cupy as cp
        return cp.asnumpy(uniq_margin(cp.asarray(self.stage._sgm.cost_volume()), cp))

    def distinctiveness(self):
        """X-26 (registered 18:28): delta(p) = mean_d C(p, d) - min_d C(p, d) of SGM32's RAW census cost volume of the
        last computed pair, over the in-range disparities d <= x (SGM32 sets out-of-range costs to the census maximum
        80, so the in-range sum = total sum - 80 x (#out-of-range)). -> float16 (H, W)."""
        sgm = self.stage._sgm
        C = sgm.cost_volume()                                    # (H, W, 128) uint8, public SGM32 method
        H, W, D = C.shape
        tot = C.sum(axis=2, dtype=np.uint32).astype(np.float64)
        n_in = np.minimum(np.arange(W) + 1, D).astype(np.float64)[None, :]
        mean_in = (tot - 80.0 * (D - n_in)) / n_in
        mn = C.min(axis=2).astype(np.float64)
        return np.clip(mean_in - mn, 0, 80).astype(np.float16)

    def stats(self):
        return self.stage.stats()


# ------------------------------------------------------------------------------------------------ gravity from the IMU
ACCEL_UP_WINDOW_S = 1.0          # class D: the first second of the run
ACCEL_UP_NORM_TOL = 0.05         # class D: |mean a| within 5 % of g (unaccelerated)
ACCEL_UP_SD_MAX = 1.0            # class D: m/s^2, per-sample spread of |a| in the window
G0 = 9.80665


def accel_up(ds, run):
    """Up (unit, in w) from the accelerometer mean over the first ACCEL_UP_WINDOW_S of the run: a_b (body = IMU frame,
    calib T_body_cam0 maps cam0 -> body) -> rectified camera: a_c = R1 R_body_cam0^T a_b -> w by the first published
    pose R_wc. -> (u, rule) or None when the checks fail or no IMU exists."""
    imu = ds.imu()
    if not len(imu.t) or not run.frames:
        return None
    f0 = run.frames[0]
    t0 = run.t[f0]
    m = (imu.t >= t0) & (imu.t <= t0 + ACCEL_UP_WINDOW_S)
    if m.sum() < 10:
        return None
    a = imu.accel[m]
    nrm = np.linalg.norm(a, axis=1)
    am = a.mean(0)
    if abs(np.linalg.norm(am) / G0 - 1.0) > ACCEL_UP_NORM_TOL or float(nrm.std()) > ACCEL_UP_SD_MAX:
        return None
    R1 = np.asarray(ds.rectify_maps()[5], float)
    Rbc = np.asarray(ds.calib["T_body_cam0"], float)[:3, :3]
    a_c = R1 @ Rbc.T @ am
    u = run.T[f0][:3, :3] @ (a_c / np.linalg.norm(a_c))
    cam_y = run.T[f0][:3, :3] @ np.array([0.0, -1.0, 0.0])
    ang = float(np.degrees(np.arccos(np.clip(u @ cam_y, -1, 1))))
    return u / np.linalg.norm(u), ("accelerometer mean over the first %.1f s (n %d, |a| %.3f m/s^2, sd %.3f), rectified "
                                   "camera via R1 R_body_cam0^T, into w by the first pose; %.2f deg from the first camera -y "
                                   "(the engine did not admit the IMU; internal CHANGES X-12)" % (ACCEL_UP_WINDOW_S, int(m.sum()),
                                                                                 float(np.linalg.norm(am)), float(nrm.std()), ang))
