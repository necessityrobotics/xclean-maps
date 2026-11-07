# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Per-keyframe camera pose refinement during map training (phase 2c arm, lead GO 2026-10-22; owner: "our xsesto is not
perfect").

Each TRAINING keyframe i gets a 6-DoF correction d_i = (w_i [rad], t_i [m]) in its own camera frame, starting at 0:
    T_wc' = T_wc @ [R(w_i) | t_i]        (R = Rodrigues),   so   T_cw' = [R^T | -R^T t] @ T_cw.
- gradients come from the photometric / depth / coverage losses through gsplat's viewmat backward;
- an L2 pull towards xsesto's pose: 0.5 * w_prior * (|w_deg|^2 / s_rot^2 + |t|^2 / s_trans^2);
- a per-row Adam (only the visited row moves: rows not seen this step keep their state and value), learning rate
  annealed linearly to 10 % at S;
- a HARD CAP (|w| <= cap_rot_deg, |t| <= cap_trans_m): a row that reaches it is a POSE FAULT, reported, never absorbed
  further;
- held-out keyframes are NEVER refined (they are scored at xsesto's pose).
All numbers are logged (report()): per keyframe, medians, maxima, cap hits and a drift-trend check along the route.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def rodrigues(w):
    """(3,) axis-angle (rad) -> (3, 3) rotation, differentiable, safe at 0."""
    th2 = (w * w).sum()
    th = torch.sqrt(th2 + 1e-24)
    K = torch.zeros(3, 3, dtype=w.dtype, device=w.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -w[2], w[1], w[2], -w[0], -w[1], w[0]
    small = th2 < 1e-12
    a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / (th2 + 1e-24))
    return torch.eye(3, dtype=w.dtype, device=w.device) + a * K + b * (K @ K)


def delta_inv(d):
    """[R | t]^-1 as a 4x4 tensor (differentiable)."""
    R = rodrigues(d[:3])
    T = torch.eye(4, dtype=d.dtype, device=d.device)
    T[:3, :3] = R.T
    T[:3, 3] = -(R.T @ d[3:])
    return T


class PoseRefiner:
    def __init__(self, n, device, start_step, total_steps, lr_rot_deg=0.05, lr_trans_m=0.005, w_prior=1e-3,
                 s_rot_deg=0.5, s_trans_m=0.05, cap_rot_deg=2.0, cap_trans_m=0.20, betas=(0.9, 0.999), eps=1e-12):
        self.n, self.device = int(n), torch.device(device)
        self.start, self.S = int(start_step), int(total_steps)
        self.lr_rot, self.lr_trans = math.radians(lr_rot_deg), float(lr_trans_m)
        self.w_prior, self.s_rot, self.s_trans = float(w_prior), math.radians(s_rot_deg), float(s_trans_m)
        self.cap_rot, self.cap_trans = math.radians(cap_rot_deg), float(cap_trans_m)
        self.b1, self.b2, self.eps = betas[0], betas[1], eps
        self.delta = torch.zeros(self.n, 6, device=self.device)
        self.m = torch.zeros(self.n, 6, device=self.device)
        self.v = torch.zeros(self.n, 6, device=self.device)
        self.k = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.cap_hit = torch.zeros(self.n, dtype=torch.bool, device=self.device)

    def active(self, step):
        return step >= self.start

    def apply(self, ti, T_cw):
        """-> (T_cw', d) with d the leaf tensor that collects the gradient of row ti."""
        d = self.delta[ti].clone().requires_grad_(True)
        return delta_inv(d) @ T_cw, d

    def R_cam(self, ti):
        return rodrigues(self.delta[ti, :3])

    @torch.no_grad()
    def update(self, ti, grad, step):
        if grad is None:
            return
        d = self.delta[ti]
        g = grad.clone()
        g[:3] += self.w_prior * d[:3] / (self.s_rot ** 2)          # prior in rad (s_rot is in rad)
        g[3:] += self.w_prior * d[3:] / (self.s_trans ** 2)
        self.k[ti] += 1
        k = float(self.k[ti])
        self.m[ti] = self.b1 * self.m[ti] + (1 - self.b1) * g
        self.v[ti] = self.b2 * self.v[ti] + (1 - self.b2) * g * g
        mh = self.m[ti] / (1 - self.b1 ** k)
        vh = self.v[ti] / (1 - self.b2 ** k)
        frac = 1.0 - 0.9 * min(max((step - self.start) / max(self.S - self.start, 1), 0.0), 1.0)
        lr = torch.tensor([self.lr_rot] * 3 + [self.lr_trans] * 3, device=self.device) * frac
        d = d - lr * mh / (vh.sqrt() + self.eps)
        r, t = d[:3].norm(), d[3:].norm()
        hit = False
        if r > self.cap_rot:
            d[:3] = d[:3] * (self.cap_rot / r)
            hit = True
        if t > self.cap_trans:
            d[3:] = d[3:] * (self.cap_trans / t)
            hit = True
        if hit:
            self.cap_hit[ti] = True
        self.delta[ti] = d

    def refined_T_wc(self, ti, T_wc):
        D = np.eye(4)
        D[:3, :3] = self.R_cam(ti).detach().cpu().numpy()
        D[:3, 3] = self.delta[ti, 3:].detach().cpu().numpy()
        return np.asarray(T_wc) @ D

    def state_dict(self):
        return dict(delta=self.delta.cpu().clone(), m=self.m.cpu().clone(), v=self.v.cpu().clone(), k=self.k.cpu().clone(),
                    cap_hit=self.cap_hit.cpu().clone())

    def load_state_dict(self, s):
        for name in ("delta", "m", "v", "k", "cap_hit"):
            setattr(self, name, s[name].to(self.device))

    def report(self, kf_ids=None, ts=None):
        d = self.delta.detach().cpu().numpy()
        rot = np.degrees(np.linalg.norm(d[:, :3], axis=1))
        tr = np.linalg.norm(d[:, 3:], axis=1)
        idx = np.arange(self.n)
        def trend(x):
            if self.n < 3 or np.std(x) == 0:
                return dict(spearman=0.0, slope_per_route=0.0)
            rx = np.argsort(np.argsort(idx)); ry = np.argsort(np.argsort(x))
            sp = float(np.corrcoef(rx, ry)[0, 1])
            sl = float(np.polyfit(idx / max(self.n - 1, 1), x, 1)[0])
            return dict(spearman=sp, slope_per_route=sl)
        comp = {n_: d[:, i] for i, n_ in enumerate(("rx_rad", "ry_rad", "rz_rad", "tx_m", "ty_m", "tz_m"))}
        return dict(
            n_refined=int(self.n), cap_rot_deg=math.degrees(self.cap_rot), cap_trans_m=self.cap_trans,
            rot_deg_median=float(np.median(rot)), rot_deg_p90=float(np.percentile(rot, 90)), rot_deg_max=float(rot.max()),
            trans_m_median=float(np.median(tr)), trans_m_p90=float(np.percentile(tr, 90)), trans_m_max=float(tr.max()),
            n_cap_hits=int(self.cap_hit.sum()), cap_hit_share=float(self.cap_hit.float().mean()),
            cap_hit_kf=[int(kf_ids[i]) if kf_ids is not None else int(i) for i in np.nonzero(self.cap_hit.cpu().numpy())[0]],
            trend_rot_deg=trend(rot), trend_trans_m=trend(tr),
            trend_components={k_: trend(v_) for k_, v_ in comp.items()},
            mean_components={k_: float(np.mean(v_)) for k_, v_ in comp.items()},
            per_kf=[dict(kf=int(kf_ids[i]) if kf_ids is not None else int(i), t=(float(ts[i]) if ts is not None else None),
                         rot_deg=float(rot[i]), trans_m=float(tr[i]), d=[float(x) for x in d[i]], cap=bool(self.cap_hit[i]))
                    for i in range(self.n)])
