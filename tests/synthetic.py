"""A tiny synthetic scene for CPU tests: a textured ground plane + a back wall, 16 cameras moving forward, 32x24 px.

Ground-truth images, disparity and masks are rendered with the torch reference rasteriser from a 'true' gaussian set;
the trainer starts from perturbed seeds. The object mimics prep_row.Prepared (same attributes) so the real Trainer,
eval and export code run unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from xclean_maps import io_lib as IO
from xclean_maps import losses as LS
from xclean_maps.render import reference_renderer


def _T_wc(z, x=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T[:3, 3] = [x, 0.0, z]
    return T


def true_scene(n_ground=220, n_wall=90, seed=0):
    rng = np.random.default_rng(seed)
    gx = rng.uniform(-2.5, 2.5, n_ground)
    gz = rng.uniform(0.5, 7.0, n_ground)
    ground = np.stack([gx, np.full(n_ground, 1.0), gz], 1)
    wx = rng.uniform(-3.0, 3.0, n_wall)
    wy = rng.uniform(-1.5, 1.0, n_wall)
    wall = np.stack([wx, wy, np.full(n_wall, 7.5)], 1)
    X = np.concatenate([ground, wall]).astype(np.float32)
    col = rng.uniform(0.1, 0.9, (len(X), 3)).astype(np.float32)
    N = len(X)
    params = dict(means=torch.tensor(X), scales=torch.full((N, 3), float(np.log(0.25))),
                  quats=torch.tensor(np.tile([1.0, 0, 0, 0], (N, 1)), dtype=torch.float32),
                  opacities=torch.full((N,), 2.5), sh0=torch.tensor((col - 0.5) / IO.SH_C0)[:, None, :],
                  shN=torch.zeros((N, 3, 3)))
    return params


class SyntheticPrepared:
    def __init__(self, n_views=16, W=32, H=24, channels=3, seed=0, row="synthetic", steps=200, cap=600):
        self.row = row
        self.W, self.H, self.channels = W, H, channels
        self.K = np.array([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1.0]])
        self.fB = 10.0
        self.up = np.array([0.0, -1.0, 0.0])
        rend = reference_renderer(antialiased=True)
        truth = true_scene(seed=seed)
        self.truth = truth
        Twc = [_T_wc(0.15 * i, x=0.05 * np.sin(i), yaw=0.03 * np.cos(i)) for i in range(n_views)]
        imgs, disps, masks = [], [], []
        Kt = torch.tensor(self.K, dtype=torch.float32)
        for T in Twc:
            with torch.no_grad():
                col, al, _ = rend(truth, torch.tensor(IO.inv_T(T), dtype=torch.float32), Kt, W, H, 0, depth=True)
            rgb = col[..., :3].clamp(0, 1).numpy()
            ed = col[..., 3].numpy()
            a = al[..., 0].numpy()
            d = np.where((a > 0.5) & (ed > 0.05), self.fB / np.maximum(ed, 1e-6), 0.0).astype(np.float32)
            img = (rgb * 255 + 0.5).astype(np.uint8)
            if channels == 1:
                img = img.mean(-1, keepdims=True).astype(np.uint8)
            m = np.where(d > 0.5, LS.BIT_VALID, 0).astype(np.uint8)
            m |= np.where(a < 0.05, LS.BIT_SKY, 0).astype(np.uint8)
            imgs.append(img)
            disps.append(d.astype(np.float16))
            masks.append(m)
        self.img = np.stack(imgs)
        self.disp = np.stack(disps)
        self.mask = np.stack(masks)
        self.views = []
        for j, T in enumerate(Twc):
            role = "hold" if j % 5 == 0 else "train"
            self.views.append(dict(j=j, kf_id=100 + j, frame_index=10 * j, t=0.1 * j, T_wc=T.tolist(), role=role, probe=role == "hold"))
        self.train_ids = [v["j"] for v in self.views if v["role"] == "train"]
        self.hold_ids = [v["j"] for v in self.views if v["role"] == "hold"]
        self.probe_ids = list(self.hold_ids)
        rng = np.random.default_rng(seed + 1)
        X = truth["means"].numpy() + rng.normal(0, 0.15, truth["means"].shape).astype(np.float32)
        X = X[: int(0.6 * len(X))]
        self.seeds = dict(xyz=X.astype(np.float32), rgb=rng.uniform(0.3, 0.7, (len(X), 3)).astype(np.float32),
                          log_scale=np.full(len(X), np.log(0.2), np.float32), source=np.ones(len(X), np.uint8))
        self.plan = dict(L=4.0, cap_max=int(cap), steps=int(steps), config_hash="synthetic-%d" % seed, row=row, near_clip_m=0.3)

    def T_wc(self, j):
        return np.array(self.views[j]["T_wc"], float)
