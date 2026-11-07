# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""`xclean-maps view <map dir>`: a simple local fly-through window (OpenCV) for a map built by xclean-maps.

Rendering:
  gsplat  (default when PyTorch sees a CUDA GPU and gsplat is installed): the trainer's own rasteriser (render.py)
  points  (fallback, CPU): every splat drawn as one coloured point with a z-buffer (fast, approximate: no sizes, no
          blending). Use it to look at a map on a machine without a GPU.
Keys: W/S forward/back, A/D left/right, Q/E down/up, drag = look, +/- speed, SPACE = play the camera path,
TAB = colour / depth, R = back to start, P = save a screenshot into the map folder, ESC = quit.
The fly-through logic follows the CPU splat script's viewer (render_xclean_splat.py viewer()).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

SH_C0 = 0.28209479177387814


def load_map(md):
    """-> dict(params (numpy, trainer layout), cams (list of 4x4 T_world_cam), K, W, H, up)."""
    from .io_lib import params_from_splat_ply
    ply = os.path.join(md, "splats.ply")
    if not os.path.isfile(ply):
        raise SystemExit("%s has no splats.ply (build the map first)" % md)
    params = params_from_splat_ply(ply)
    cams, K, W, H, up = [], None, 960, 600, None
    cj = os.path.join(md, "cameras.json")
    if os.path.isfile(cj):
        with open(cj, encoding="utf-8") as f:
            cl = json.load(f)
        cams = [np.array(c["T_world_cam"], float) for c in cl]
        if cl:
            c = cl[0]
            K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
            W, H = int(c["width"]), int(c["height"])
    rj = os.path.join(md, "report.json")
    if os.path.isfile(rj):
        with open(rj, encoding="utf-8") as f:
            up = (json.load(f) or {}).get("up")
    if not cams:                                  # no camera file: look at the map centre from outside
        c = np.median(params["means"], 0)
        ext = float(np.percentile(np.linalg.norm(params["means"] - c, axis=1), 90))
        from .render import look_at
        cams = [np.linalg.inv(look_at(c + np.array([0, -ext, -ext]), c, np.array([0, -1.0, 0])))]
    if K is None:
        K = np.array([[0.8 * W, 0, W / 2], [0, 0.8 * W, H / 2], [0, 0, 1.0]])
    if up is None:
        up = np.mean([T[:3, :3] @ np.array([0.0, -1.0, 0.0]) for T in cams], 0)
    up = np.asarray(up, float) / np.linalg.norm(up)
    P = np.array([T[:3, 3] for T in cams])
    extent = float(np.linalg.norm(P.max(0) - P.min(0))) if len(P) > 1 else 10.0
    return dict(params=params, cams=cams, K=K, W=W, H=H, up=up, extent=max(extent, 1.0))


class PointRenderer:
    """CPU fallback: splat centres as points, nearest wins (no blending, no sizes)."""
    kind = "points (CPU, approximate)"

    def __init__(self, params, max_points=1_500_000):
        a = 1.0 / (1.0 + np.exp(-params["opacities"].reshape(-1)))
        keep = a > 0.2
        X = params["means"][keep]
        col = np.clip(SH_C0 * params["sh0"][keep, 0, :] + 0.5, 0, 1)
        if len(X) > max_points:
            sel = np.random.RandomState(0).choice(len(X), max_points, replace=False)
            X, col = X[sel], col[sel]
        self.X, self.col = X.astype(np.float32), (col * 255).astype(np.uint8)

    def render(self, T_wc, K, W, H):
        R, t = T_wc[:3, :3].T, -T_wc[:3, :3].T @ T_wc[:3, 3]
        Xc = self.X @ R.T + t
        z = Xc[:, 2]
        ok = z > 0.05
        u = np.round(K[0, 0] * Xc[ok, 0] / z[ok] + K[0, 2]).astype(np.int64)
        v = np.round(K[1, 1] * Xc[ok, 1] / z[ok] + K[1, 2]).astype(np.int64)
        zz, cc = z[ok], self.col[ok]
        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, zz, cc = u[ins], v[ins], zz[ins], cc[ins]
        o = np.argsort(-zz)                                  # far first, near overwrites
        img = np.full((H, W, 3), 12, np.uint8)
        dep = np.zeros((H, W), np.float32)
        img[v[o], u[o]] = cc[o]
        dep[v[o], u[o]] = zz[o]
        return img, dep


class GsplatViewRenderer:
    kind = "gsplat (GPU)"

    def __init__(self, params):
        import torch
        from .render import gsplat_renderer
        self.torch = torch
        self.p = {k: torch.as_tensor(np.ascontiguousarray(v), device="cuda") for k, v in params.items()}
        self.r = gsplat_renderer("antialiased", False, 0.05, 1000.0, 0.3)

    def render(self, T_wc, K, W, H):
        torch = self.torch
        with torch.no_grad():
            Tcw = torch.as_tensor(np.linalg.inv(T_wc), dtype=torch.float32, device="cuda")
            Kt = torch.as_tensor(K, dtype=torch.float32, device="cuda")
            col, al, _ = self.r(self.p, Tcw, Kt, W, H, 1, depth=True)
            rgb = (col[..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
            dep = col[..., 3].cpu().numpy() * (al[..., 0].cpu().numpy() > 0.3)
        return np.ascontiguousarray(rgb), dep


def make_renderer(params, mode="auto"):
    if mode in ("auto", "gsplat"):
        try:
            import torch
            if torch.cuda.is_available():
                return GsplatViewRenderer(params)
            if mode == "gsplat":
                raise SystemExit("--renderer gsplat: PyTorch sees no CUDA GPU")
        except ImportError as e:
            if mode == "gsplat":
                raise SystemExit("--renderer gsplat: %s" % e)
    return PointRenderer(params)


def depth_colour(dep):
    import cv2
    inv = np.where(dep > 0, 1.0 / np.maximum(dep, 1e-3), 0.0)
    m = inv > 0
    v = np.clip(inv / max(float(np.percentile(inv[m], 98)) if m.any() else 1.0, 1e-6), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


class FlyState:
    """Camera state of the fly-through (yaw / pitch about the map's up vector)."""

    def __init__(self, cams, up, extent):
        self.cams, self.up = cams, up
        f0 = cams[0][:3, 2]
        h0 = f0 - up * (f0 @ up)
        if np.linalg.norm(h0) < 1e-6:
            h0 = np.cross(up, [1.0, 0, 0])
        self.h0 = h0 / np.linalg.norm(h0)
        self.speed = extent * 0.02
        self.reset()

    def reset(self):
        self.pos = self.cams[0][:3, 3].copy()
        self.yaw = 0.0
        self.pitch = math.asin(max(-1.0, min(1.0, float(self.cams[0][:3, 2] @ self.up))))
        self.play = -1

    def pose(self):
        up, h0 = self.up, self.h0
        hf = h0 * math.cos(self.yaw) - np.cross(up, h0) * math.sin(self.yaw)
        f = hf * math.cos(self.pitch) + up * math.sin(self.pitch)
        right = np.cross(f, up)
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        T = np.eye(4)
        T[:3, :3] = np.stack([right, down, f], 1)
        T[:3, 3] = self.pos
        return T, f, right

    def key(self, k):
        T, f, right = self.pose()
        mv = {ord("w"): f, ord("s"): -f, ord("d"): right, ord("a"): -right, ord("e"): self.up, ord("q"): -self.up}
        if k in mv:
            self.pos = self.pos + mv[k] * self.speed
        elif k in (ord("+"), ord("=")):
            self.speed *= 1.5
        elif k == ord("-"):
            self.speed /= 1.5
        elif k == ord(" "):
            self.play = 0 if self.play < 0 else -1
        elif k == ord("r"):
            self.reset()


def render_frame(rend, st, m, mode="colour"):
    if st.play >= 0:
        T = st.cams[st.play]
        st.play = st.play + 1 if st.play + 1 < len(st.cams) else -1
    else:
        T = st.pose()[0]
    img, dep = rend.render(T, m["K"], m["W"], m["H"])
    return img[:, :, ::-1].copy() if mode == "colour" else depth_colour(dep)


def view(md, mode="auto", snapshot=None):
    """Open the window, or (snapshot=path) render the first camera once to a PNG and return (no window)."""
    import cv2
    m = load_map(md)
    rend = make_renderer(m["params"], mode)
    st = FlyState(m["cams"], m["up"], m["extent"])
    if snapshot:
        fr = render_frame(rend, st, m)
        ok, buf = cv2.imencode(".png", fr)
        with open(snapshot, "wb") as f:
            f.write(buf.tobytes())
        return dict(renderer=rend.kind, n_splats=int(len(m["params"]["means"])), size=[m["W"], m["H"]], file=snapshot)
    title = "xclean-maps: %s (%s)" % (os.path.basename(os.path.abspath(md)), rend.kind)
    drag = {"p": None}

    def mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            drag["p"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            drag["p"] = None
        elif ev == cv2.EVENT_MOUSEMOVE and drag["p"]:
            dx, dy = x - drag["p"][0], y - drag["p"][1]
            st.yaw += dx * 0.005
            st.pitch = max(-1.5, min(1.5, st.pitch - dy * 0.005))
            drag["p"] = (x, y)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    print("viewer (%s): W/S forward/back, A/D left/right, Q/E down/up, drag = look, +/- speed, SPACE = play the path, "
          "TAB = colour/depth, R = start, P = screenshot, ESC = quit" % rend.kind, flush=True)
    vmode = "colour"
    while True:
        t0 = time.time()
        fr = render_frame(rend, st, m, vmode)
        cv2.putText(fr, "%.1f fps  speed %.2f m  %s" % (1 / max(1e-3, time.time() - t0), st.speed,
                                                         "PLAYING" if st.play >= 0 else ""),
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(title, fr)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == 9:
            vmode = "depth" if vmode == "colour" else "colour"
        elif k == ord("p"):
            fn = os.path.join(md, "screenshot_%s.png" % time.strftime("%Y%m%d_%H%M%S"))
            cv2.imwrite(fn, fr)
            print("saved %s" % fn, flush=True)
        elif k != 255:
            st.key(k)
    cv2.destroyAllWindows()
    return dict(renderer=rend.kind)
