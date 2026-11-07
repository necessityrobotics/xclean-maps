# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Live view of the map build (owner 2026-10-18; lead A5).

Every N training steps the trainer renders the CURRENT model with its own rasteriser (same process, same CUDA
context, torch.no_grad) from 3 fixed views and writes <work>/<row>/live/preview.jpg + live/stats.json atomically
(temp file + os.replace, with retries while a reader holds the file). N is re-chosen after every preview from the
measured cost so the overhead stays <= 3 % of the training step time; the measured overhead is written with the stats.

Views: (1) a held-out keyframe beside its real photo; (2) an overview from above the route; (3) the latest trained
keyframe beside its photo. Nothing here can stop the build: every failure is logged and swallowed.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import util as U


def _to_u8(x):
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _fit_w(img, w):
    import cv2
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = w / img.shape[1]
    return cv2.resize(img, (int(w), max(1, int(round(img.shape[0] * s)))), interpolation=cv2.INTER_AREA)


def _label(img, text):
    import cv2
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 235, 240), 1, cv2.LINE_AA)
    return out


def overview_camera(cam_positions, up, K_like, W=960, H=540, fov_deg=60.0):
    """A fixed camera above the route centre looking down along -up, framing every keyframe centre."""
    from .render import look_at
    P = np.asarray(cam_positions, float)
    up = np.asarray(up, float) / np.linalg.norm(up)
    c = P.mean(0)
    h = P - c - np.outer((P - c) @ up, up)
    ext = float(np.linalg.norm(h, axis=1).max()) if len(P) > 1 else 5.0
    f = 0.5 * W / math.tan(math.radians(fov_deg) / 2)
    dist = max(ext / math.tan(math.radians(fov_deg) / 2) * 1.15, 5.0)
    eye = c + up * dist
    # an image-up direction in the ground plane: the route's main horizontal axis
    a = np.linalg.svd(h, full_matrices=False)[2][0] if len(P) > 2 else np.array([1.0, 0, 0])
    a = a - (a @ up) * up
    if np.linalg.norm(a) < 1e-6:
        a = np.array([1.0, 0, 0]) - up[0] * up
    a /= np.linalg.norm(a)
    T = look_at(eye, c, a)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return T, K, W, H


class LivePreview:
    def __init__(self, live_dir, row, stage, renderer, K, W, H, hold_view, overview, every_init=250, every_min=50,
                 every_max=5000, overhead_max=0.03, width=1600, log=None, device="cpu"):
        self.dir = U.ensure_dir(live_dir)
        self.row, self.stage = row, stage
        self.render = renderer
        self.K, self.W, self.H = K, int(W), int(H)
        self.hold = hold_view                 # dict(j, kf_id, img (H,W,3 float), T_cw np)
        self.ov = overview                    # (T_cw, K, W, H)
        self.every = int(every_init)
        self.every_min, self.every_max = int(every_min), int(every_max)
        self.overhead_max = float(overhead_max)
        self.width = int(width)
        self.log = log or (lambda m: None)
        self.device = device
        self.last_step = -10 ** 9
        self.cost_s = None
        self.overhead_frac = None
        self.n_written = 0
        self.n_failed = 0

    def due(self, step):
        return step - self.last_step >= self.every

    def _render(self, params, T_cw, K, W, H, sh_deg, sky=None):
        import torch
        with torch.no_grad():
            Tt = torch.as_tensor(T_cw, dtype=torch.float32, device=self.device)
            Kt = torch.as_tensor(K, dtype=torch.float32, device=self.device)
            col, al, _ = self.render(params, Tt, Kt, W, H, sh_deg, depth=False)
            rgb = col[..., :3]
            if sky is not None:
                from .sky import ray_dirs_world
                from .io_lib import inv_T
                rgb = rgb + (1 - al) * sky.sample(ray_dirs_world(inv_T(np.asarray(T_cw)), K, W, H, self.device))
            return rgb.clamp(0, 1).cpu().numpy()

    def write(self, step, total, stats, params, sh_deg, sky, latest, step_time_s):
        """latest: dict(j, kf_id, img (H,W,3 float numpy), T_cw np). Never raises."""
        t0 = time.perf_counter()
        try:
            import cv2
            r_hold = self._render(params, self.hold["T_cw"], self.K, self.W, self.H, sh_deg, sky)
            T_ov, K_ov, W_ov, H_ov = self.ov
            r_ov = self._render(params, T_ov, K_ov, W_ov, H_ov, sh_deg, None)
            r_lat = self._render(params, latest["T_cw"], self.K, self.W, self.H, sh_deg, sky)
            half = self.width // 2
            bgr = lambda a: _to_u8(a)[:, :, ::-1]                                        # noqa: E731
            row1 = np.hstack([_label(_fit_w(bgr(r_hold), half), "render, held-out kf %d (never trained on)" % self.hold["kf_id"]),
                              _label(_fit_w(bgr(self.hold["img"]), half), "photo, held-out kf %d" % self.hold["kf_id"])])
            row2 = _label(_fit_w(bgr(r_ov), self.width), "overview from above the route (current model, no sky)")
            row3 = np.hstack([_label(_fit_w(bgr(r_lat), half), "render, latest trained kf %d" % latest["kf_id"]),
                              _label(_fit_w(bgr(latest["img"]), half), "photo, kf %d" % latest["kf_id"])])
            img = np.vstack([row1, row2, row3])
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("jpeg encode failed")
            U.atomic_write_bytes(os.path.join(self.dir, "preview.jpg"), enc.tobytes(), retries=40)
            cost = time.perf_counter() - t0
            self.cost_s = cost
            if step_time_s and step_time_s > 0:
                need = int(math.ceil(cost / (self.overhead_max * step_time_s)))
                self.every = int(min(max(need, self.every_min), self.every_max))
                self.overhead_frac = cost / (self.every * step_time_s)
            st = dict(stats, row=self.row, stage=self.stage, step=int(step), total_steps=int(total),
                      preview_every=self.every, preview_cost_s=round(cost, 4),
                      preview_overhead_frac=None if self.overhead_frac is None else round(self.overhead_frac, 5),
                      written=time.strftime("%Y-%m-%dT%H:%M:%S"), preview="preview.jpg")
            U.atomic_write_text(os.path.join(self.dir, "stats.json"), json.dumps(st, indent=1, default=U.np_default), retries=40)
            self.n_written += 1
        except Exception as e:   # the live view must never affect the build
            self.n_failed += 1
            self.log("live preview failed at step %d (ignored): %r" % (step, e))
        self.last_step = step
        return self.cost_s
