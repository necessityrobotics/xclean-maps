# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Sky panorama (BUILDER 5.7): an equirectangular texture sampled by WORLD view direction, composited behind the
gaussians as C = C_gauss + (1 - alpha) C_sky (manual compositing: gsplat's `backgrounds` is a per-camera constant).

The panorama is exported as maps/<row>/sky_equirect.png only; it is NOT baked into the splat files and the current
viewer is not changed (showing a learned sky is the owner's call, G-j). With alpha = 1 everywhere the compositing is a
no-op (test 0b(7)).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def world_basis(up):
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return np.stack([e1, e2, up], 0)            # rows: e1, e2, up


class SkyPanorama:
    def __init__(self, up, channels=3, W=1024, H=512, device="cpu", init=0.0):
        self.basis = torch.tensor(world_basis(up), dtype=torch.float32, device=device)
        self.channels = int(channels)
        self.param = torch.nn.Parameter(torch.full((1, self.channels, int(H), int(W)), float(init),
                                                   dtype=torch.float32, device=device))

    def sample(self, dirs_w):
        """dirs_w [H, W, 3] world ray directions -> sky colour [H, W, 3] (grey replicated)."""
        d = F.normalize(dirs_w, dim=-1) @ self.basis.T                       # coordinates in (e1, e2, up)
        lon = torch.atan2(d[..., 1], d[..., 0])                              # [-pi, pi]
        lat = torch.asin(torch.clamp(d[..., 2], -1.0, 1.0))                  # [-pi/2, pi/2]
        W = self.param.shape[-1]
        u = lon / np.pi
        # circular in longitude (review N5b): pad one texel column from the opposite side and remap x accordingly
        tex = torch.cat([self.param[..., -1:], self.param, self.param[..., :1]], -1)
        u = ((u + 1.0) * W + 2.0) / (W + 2.0) - 1.0
        grid = torch.stack([u, -lat / (np.pi / 2)], -1)[None]              # x right, y down
        s = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=False)
        s = torch.clamp(s[0].permute(1, 2, 0), 0.0, 1.0)
        if self.channels == 1:
            s = s.expand(-1, -1, 3)
        mn = getattr(self, "min_lat_deg", None)
        if mn is not None:
            # sky v2 (lead 2026-10-22): the panorama is the SKY only - black below min_lat_deg, so ground is never painted
            # by it (in training it took ground texture and let low-alpha ground through)
            s = s * (lat >= np.radians(mn)).to(s.dtype)[..., None]
        return s

    def observed_mask(self, dirs_list):
        """Texels any of the given ray-direction sets [H, W, 3] fall in (an observed-texel map for sky_equirect.json)."""
        Hs, Ws = self.param.shape[-2:]
        obs = np.zeros((Hs, Ws), bool)
        for d in dirs_list:
            q = F.normalize(d, dim=-1) @ self.basis.T
            lon = torch.atan2(q[..., 1], q[..., 0]).cpu().numpy()
            lat = torch.asin(torch.clamp(q[..., 2], -1.0, 1.0)).cpu().numpy()
            x = np.clip(((lon / np.pi + 1) * 0.5 * Ws).astype(int), 0, Ws - 1)
            y = np.clip(((-lat / (np.pi / 2) + 1) * 0.5 * Hs).astype(int), 0, Hs - 1)
            obs[y.ravel(), x.ravel()] = True
        return obs

    def to_png_array(self):
        a = self.param.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        a = (a * 255.0 + 0.5).astype(np.uint8)
        return a[..., 0] if self.channels == 1 else a


def ray_dirs_world(T_wc, K, W, H, device="cpu"):
    """World-frame ray directions of every pixel (OpenCV convention: pixel (i, j) centred at image coordinates (j, i),
    the grid of the rectified photos and of K; review F5 - the phase-1 '+0.5' put the sky half a pixel off)."""
    T = torch.as_tensor(T_wc, dtype=torch.float32, device=device)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    jj, ii = torch.meshgrid(torch.arange(W, device=device, dtype=torch.float32),
                            torch.arange(H, device=device, dtype=torch.float32), indexing="xy")
    x = (jj - Kt[0, 2]) / Kt[0, 0]
    y = (ii - Kt[1, 2]) / Kt[1, 1]
    d = torch.stack([x, y, torch.ones_like(x)], -1)
    return d @ T[:3, :3].T


def composite(colors_rgb, alphas, sky_rgb):
    """C = C_gauss (premultiplied) + (1 - alpha) * C_sky."""
    return colors_rgb + (1.0 - alphas) * sky_rgb
