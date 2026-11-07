# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""I/O shared by PREP, the trainer, eval and export. Copied and adapted from the old builder (BUILDER 12.2):
[GS] gs_train.py read_cache_bin / rect_maps / load_rect_image / ssim; [SL] showcase_lib.py PLY writers, read_kit_traj.

Changes from the originals (all deliberate, all tested in tests/):
  * disparity is kept as DISPARITY (fp16 at storage), not converted to depth (depth loss is in disparity space);
  * grey images stay ONE channel (load_rect_image(..., channels=1)); colour images are RGB (BGR -> RGB as [GS]);
  * the engine cache header is returned with the array (digest per frame for V0.2 provenance);
  * ssim works on any device and takes an optional mask.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

from . import util as U

# the engine's own cache header (xsesto/stages/cache.py: magic, ver, dtype_code, h, w, digest)
CACHE_HDR = struct.Struct("<4sHHIIQ")
CACHE_MAGIC = b"XONE"
_DTYPES = {0: np.float32, 1: np.float16, 2: np.uint8, 3: np.int32}
SH_C0 = 0.28209479177387814


# --------------------------------------------------------------------------------------------- depth cache
def read_cache_header(path):
    with open(path, "rb") as f:
        head = f.read(CACHE_HDR.size)
    magic, ver, code, h, w, dig = CACHE_HDR.unpack(head)
    return dict(magic=magic, ver=int(ver), code=int(code), h=int(h), w=int(w), digest=int(dig))


def read_cache_bin(path):
    """-> (float32 array h x w, header dict). Same parse as [GS] read_cache_bin, plus the header."""
    with open(path, "rb") as f:
        head = f.read(CACHE_HDR.size)
        magic, ver, code, h, w, dig = CACHE_HDR.unpack(head)
        if magic != CACHE_MAGIC:
            raise ValueError("%s: bad cache magic %r" % (path, magic))
        a = np.frombuffer(f.read(), dtype=_DTYPES[code])
    return a.reshape(h, w).astype(np.float32), dict(ver=int(ver), code=int(code), h=int(h), w=int(w), digest=int(dig))


def cache_path(cache_dir, frame_index):
    return os.path.join(cache_dir, "%06d.bin" % int(frame_index))


def load_disparity(cache_dir, frame_index, w, h):
    """RAFT disparity (px) of a keyframe from the engine cache, cropped to the image size; None when absent."""
    p = cache_path(cache_dir, frame_index)
    if not os.path.isfile(p):
        return None, None
    d, hdr = read_cache_bin(p)
    return np.ascontiguousarray(d[:h, :w]), hdr


def depth_from_disparity(disp, fB, min_disp=0.5):
    """The engine's dense depth model (BUILDER 4.1 V0.2): z = fB / d, no offset; d <= min_disp -> 0 (invalid).
    Identical expression to [GS] load_depth."""
    return np.where(disp > min_disp, fB / np.maximum(disp, 1e-6), 0.0).astype(np.float32)


# --------------------------------------------------------------------------------------------- rectification
def rect_maps(intr, R1, M1, D1):
    """Exactly [GS] rect_maps: K_rect = [fx 0 cx; 0 fx cy], maps from the kit calibration."""
    import cv2
    fx, cx, cy, w, h = intr
    K_rect = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    m1x, m1y = cv2.initUndistortRectifyMap(M1, D1, R1, K_rect, (w, h), cv2.CV_16SC2)
    return K_rect, m1x, m1y


def load_rect_image(path, m1x, m1y, channels=3):
    """-> rectified uint8 image. channels=3: RGB (grey replicated, BGR->RGB), byte-identical to [GS] load_rect_image.
    channels=1: one grey channel (OAK / Rosario v2 IR), rectified with the same maps and interpolation."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if channels == 1:
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(cv2.remap(img, m1x, m1y, cv2.INTER_LINEAR))
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(cv2.remap(img, m1x, m1y, cv2.INTER_LINEAR)[:, :, ::-1])


def is_grey_file(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        return True
    return bool(np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]))


# --------------------------------------------------------------------------------------------- poses
def inv_T(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def q_xyzw_to_R(q):
    """TUM order (qx, qy, qz, qw), W LAST, as xsesto/kit/writer.py writes it."""
    q = np.asarray(q, np.float64).reshape(-1, 4)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def R_to_q_xyzw(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R).reshape(-1, 3, 3)).as_quat()


def read_kit_traj(kit):
    """kit/trajectory_kf.csv (frame T_world_cam0_rectified; t,x,y,z,qx,qy,qz,qw,kf_id,frame_index) -> dict, file order."""
    rows = []
    with open(os.path.join(kit, "trajectory_kf.csv"), encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            v = ln.strip().split(",")
            rows.append([float(x) for x in v[:8]] + [int(v[8]), int(v[9])])
    a = np.array([r[:8] for r in rows], float)
    return dict(t=a[:, 0], p=a[:, 1:4], q=a[:, 4:8], kf_id=np.array([r[8] for r in rows], int),
                frame_index=np.array([r[9] for r in rows], int))


def read_tum(path):
    a = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64, ndmin=2)
    return a[np.isfinite(a[:, :8]).all(1)]


def read_kit_frames(path):
    """kit/trajectory_frames.csv -> (t, frame_index, state[str], p, q). Columns from xsesto/kit/writer.py:
    t,x,y,z,qx,qy,qz,qw,frame_index,state,inliers,budget_at_cap,budget."""
    t, fi, st, P, Q = [], [], [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split(",")
            try:
                t.append(float(c[0]))
            except ValueError:
                continue
            P.append([float(c[1]), float(c[2]), float(c[3])])
            Q.append([float(c[4]), float(c[5]), float(c[6]), float(c[7])])
            fi.append(int(c[8]) if len(c) > 8 else -1)
            st.append(c[9] if len(c) > 9 else "tracking")
    t = np.array(t, float)
    o = np.argsort(t, kind="stable")
    return t[o], np.array(fi, int)[o], np.array(st, dtype=object)[o], np.array(P, float)[o], np.array(Q, float)[o]


def T_from_pq(p, q):
    T = np.eye(4)
    T[:3, :3] = q_xyzw_to_R(q)[0]
    T[:3, 3] = p
    return T


# --------------------------------------------------------------------------------------------- SSIM (torch)
def gauss_win(size=11, sigma=1.5, device="cpu"):
    import torch
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim_map(a, b, win):
    """[GS] ssim maths (11x11 Gaussian, c1 0.01^2, c2 0.03^2), returned as a per-pixel map (1, C, H, W)."""
    import torch.nn.functional as F
    C = a.shape[1]
    w = win.expand(C, 1, -1, -1)
    pad = win.shape[-1] // 2
    mu_a = F.conv2d(a, w, padding=pad, groups=C)
    mu_b = F.conv2d(b, w, padding=pad, groups=C)
    s_aa = F.conv2d(a * a, w, padding=pad, groups=C) - mu_a ** 2
    s_bb = F.conv2d(b * b, w, padding=pad, groups=C) - mu_b ** 2
    s_ab = F.conv2d(a * b, w, padding=pad, groups=C) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))


def ssim(a, b, win, mask=None):
    """Mean SSIM; with a mask (H, W bool/float) the mean over masked pixels. Without a mask == [GS] ssim."""
    m = ssim_map(a, b, win)
    if mask is None:
        return m.mean()
    mk = mask.to(m.dtype)[None, None].expand_as(m)
    return (m * mk).sum() / mk.sum().clamp_min(1.0)


def psnr_np(pred, gt, mask=None):
    """pred/gt float arrays in [0, 1]."""
    d = (np.clip(pred, 0, 1) - gt) ** 2
    if mask is not None:
        d = d[mask]
    mse = max(float(np.mean(d)), 1e-10)
    return float(-10.0 * np.log10(mse))


# --------------------------------------------------------------------------------------------- PLY
def write_points_ply(path, xyz, rgb):
    """[SL] write_points_ply (x y z float, red green blue uchar), atomic."""
    xyz = np.ascontiguousarray(xyz, np.float32)
    rgb = np.ascontiguousarray(rgb, np.uint8)
    n = len(xyz)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
           "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % n)
    U.atomic_write_bytes(path, hdr.encode("ascii") + arr.tobytes())


SPLAT_PLY_FIELDS = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + ["f_rest_%d" % i for i in range(9)] +
                    ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def write_splat_ply(path, xyz, f_dc, f_rest_chmajor, opac_logit, log_scales, quats_wxyz):
    """The INRIA 3DGS schema [SL write_splat_ply]: f_rest CHANNEL-major, opacity logit, log scales, rot_0 = w."""
    N = len(xyz)
    hdr = ["ply", "format binary_little_endian 1.0", "element vertex %d" % N,
           "property float x", "property float y", "property float z",
           "property float nx", "property float ny", "property float nz"]
    hdr += ["property float f_dc_%d" % i for i in range(3)]
    hdr += ["property float f_rest_%d" % i for i in range(f_rest_chmajor.shape[1])]
    hdr += ["property float opacity", "property float scale_0", "property float scale_1", "property float scale_2",
            "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3", "end_header"]
    arr = np.concatenate([xyz, np.zeros((N, 3), np.float32), f_dc, f_rest_chmajor, opac_logit.reshape(-1, 1),
                          log_scales, quats_wxyz], axis=1).astype(np.float32)
    U.atomic_write_bytes(path, ("\n".join(hdr) + "\n").encode("ascii") + np.ascontiguousarray(arr).tobytes())
    return N


PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
             "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
             "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}


def read_ply_vertices(path):
    """Read-only memmap of the vertex element of a binary little-endian PLY (as [BMA] read_ply_vertices)."""
    head = b""
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("%s: no end_header" % path)
            head += line
            if line.strip() == b"end_header":
                break
            if len(head) > 1 << 16:
                raise ValueError("%s: header too long" % path)
    lines = head.decode("ascii").splitlines()
    if lines[0].strip() != "ply" or "binary_little_endian" not in lines[1]:
        raise ValueError("%s: not a binary little-endian PLY" % path)
    props, n, seen = [], None, []
    for ln in lines[2:]:
        t = ln.split()
        if not t:
            continue
        if t[0] == "element":
            seen.append(t[1])
            if t[1] == "vertex":
                n = int(t[2])
        elif t[0] == "property" and seen and seen[-1] == "vertex":
            props.append((t[2], PLY_TYPES[t[1]]))
    return np.memmap(path, dtype=np.dtype(props), mode="r", offset=len(head), shape=(n,))


def params_from_splat_ply(path):
    """splats.ply -> numpy parameter dict in the trainer's layout (means, scales, quats wxyz, opacities logit,
    sh0 [N,1,3], shN [N,3,3]); the exact inverse of export's write (re-score from the file, BUILDER 7.2)."""
    v = read_ply_vertices(path)
    N = int(v.shape[0])
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    opac = np.asarray(v["opacity"], np.float32).copy()
    sh0 = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)[:, None, :]
    rest = np.stack([v["f_rest_%d" % i] for i in range(9)], 1).astype(np.float32)      # channel-major c*3 + k
    shN = rest.reshape(N, 3, 3).transpose(0, 2, 1).copy()                              # -> [N, k, c]
    return dict(means=means, scales=scales, quats=quats, opacities=opac, sh0=sh0, shN=shN)


def splat_ply_arrays(params):
    """Trainer params (numpy) -> the arrays write_splat_ply takes (as [FIN] stage_export)."""
    N = len(params["means"])
    sh = np.concatenate([params["sh0"], params["shN"]], 1)
    f_dc = sh[:, 0, :]
    f_rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(N, -1)
    q = params["quats"] / np.linalg.norm(params["quats"], axis=1, keepdims=True)
    return (params["means"].astype(np.float32), f_dc.astype(np.float32), f_rest.astype(np.float32),
            params["opacities"].reshape(-1).astype(np.float32), params["scales"].astype(np.float32), q.astype(np.float32))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
