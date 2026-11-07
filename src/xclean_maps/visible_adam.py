# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""VisibleAdam (BUILDER 5.3): Adam that updates only the rows that received a nonzero data-term gradient.

Per parameter tensor (one optimiser per tensor, one param group, as gsplat strategies require) the state is
  exp_avg, exp_avg_sq : same shape as the parameter
  row_step            : float32 [N]  (float32, not int: gsplat's sample_add appends float zeros to every state tensor;
                                      stored under a key other than "step", because gsplat's _update_param_with_optimizer
                                      skips the key "step" when it relocates / appends rows)
step(vis) updates m, v, row_step and the parameter on the rows where vis is True; every other row is untouched
(parameter, m, v and row_step byte for byte). With every row visible on every step it is torch.optim.Adam (same
lerp / addcmul / bias-correction / addcdiv order as torch's single-tensor path, bias corrections in double).
"""
from __future__ import annotations

import math

import torch


class VisibleAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-15):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, vis=None, closure=None):
        """vis: bool tensor [N] (rows to update) or None (= all rows)."""
        assert closure is None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "exp_avg" not in st or st["exp_avg"].shape != p.shape:
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["row_step"] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
                m, v, rs = st["exp_avg"], st["exp_avg_sq"], st["row_step"]
                if vis is None:
                    rows = None
                else:
                    rows = torch.nonzero(vis.reshape(-1), as_tuple=True)[0]
                    if rows.numel() == 0:
                        continue
                    if rows.numel() == p.shape[0]:
                        rows = None
                if rows is None:
                    _adam_rows(p, g, m, v, rs, lr, b1, b2, eps)
                else:
                    pr, gr, mr, vr, rsr = p[rows], g[rows], m[rows], v[rows], rs[rows]
                    _adam_rows(pr, gr, mr, vr, rsr, lr, b1, b2, eps)
                    p[rows] = pr
                    m[rows] = mr
                    v[rows] = vr
                    rs[rows] = rsr
        return None


def _adam_rows(p, g, m, v, rs, lr, b1, b2, eps):
    """In-place Adam on (p, m, v, rs) restricted to the tensors passed (already row-selected)."""
    rs.add_(1.0)
    m.lerp_(g, 1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    smin, smax = float(rs.min()), float(rs.max())
    if smin == smax:                                  # one bias correction for every row (torch's scalar path)
        step = smin
        bc1 = 1 - b1 ** step
        bc2 = 1 - b2 ** step
        step_size = lr / bc1
        bc2_sqrt = bc2 ** 0.5                         # torch: bias_correction2 ** 0.5 (not math.sqrt)
        denom = (v.sqrt() / bc2_sqrt).add_(eps)
        p.addcdiv_(m, denom, value=-step_size)
        return
    # per-row bias corrections (rows that were seen a different number of times), in double
    s = rs.to(torch.float64)
    bc1 = 1 - torch.pow(torch.tensor(b1, dtype=torch.float64, device=p.device), s)
    bc2 = 1 - torch.pow(torch.tensor(b2, dtype=torch.float64, device=p.device), s)
    shape = (-1,) + (1,) * (p.dim() - 1)
    step_size = (lr / bc1).to(p.dtype).reshape(shape)
    bc2_sqrt = torch.sqrt(bc2).to(p.dtype).reshape(shape)
    denom = (v.sqrt() / bc2_sqrt).add_(eps)
    p.sub_(step_size * (m / denom))


def make_optimizers(params, lrs, betas=(0.9, 0.999), eps=1e-15):
    """One VisibleAdam per parameter tensor ({name: optimizer}), the layout gsplat strategies expect."""
    return {n: VisibleAdam([{"params": [params[n]], "lr": lrs[n], "name": n}], lr=lrs[n], betas=betas, eps=eps)
            for n in lrs}
