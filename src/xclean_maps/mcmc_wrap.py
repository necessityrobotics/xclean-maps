# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""MCMC with a fixed budget, in metric units (BUILDER 5.2).

* noise_lr = 5e5 / L^2. gsplat adds noise = Sigma * randn * sigmoid(100 ((1 - alpha) - 0.995)) * lr_means * noise_lr with
  Sigma in m^2 [GSP strategy/ops.py inject_noise_to_position]; with lr_means = 1.6e-4 L the default 5e5 is L^2 too
  large in metres. With 5e5 / L^2 a metric run moves a gaussian by exactly L x the normalised-unit move (test 0b(4)).
* relocate() wrapper: gsplat zeroes the optimiser state of the SOURCE rows only; the DESTINATION (dead) rows keep the
  old exp_avg / exp_avg_sq / row_step. The wrapper zeroes them too.
* sample_add appends float32 zeros to every state tensor (so row_step is float32; VisibleAdam stores it under
  "row_step", not "step", which _update_param_with_optimizer skips).
* coverage re-seeding (5.2): move dead (then lowest-opacity) rows onto RAFT points where the render leaves holes,
  at most 2 % of cap_max per pass, with their optimiser state zeroed.

cpu_reference_kernels(): a context manager that swaps gsplat's two CUDA-only kernels used by these ops
(compute_relocation, quat_scale_to_covar_preci) for exact torch ports, so the real gsplat ops run on CPU tensors in the
unit tests. It is never used on the GPU path.
"""
from __future__ import annotations

import contextlib
import math

import torch
from gsplat.strategy import MCMCStrategy
from gsplat.strategy import ops as gops


def noise_lr_metric(L, base=5e5):
    return float(base) / (float(L) ** 2)


# --------------------------------------------------------------------------------------------- CPU reference kernels
def compute_relocation_torch(opacities, scales, ratios, binoms):
    """Exact port of gsplat/cuda/csrc/compute_relocation.cu (Eq. 9 of 3DGS-as-MCMC), float32 like the kernel."""
    n_max = binoms.shape[0]
    ratios = ratios.clamp(min=1, max=n_max).int()
    op = opacities.float()
    new_op = 1.0 - torch.pow(1.0 - op, 1.0 / ratios.float())
    denom = torch.zeros_like(op)
    rmax = int(ratios.max()) if ratios.numel() else 0
    for i in range(1, rmax + 1):
        active = ratios >= i
        for k in range(0, i):
            term = ((-1.0) ** k / math.sqrt(k + 1)) * torch.pow(new_op, k + 1)
            denom = denom + torch.where(active, binoms[i - 1, k].float() * term, torch.zeros_like(term))
    coeff = op / denom
    return new_op, coeff[:, None] * scales.float()


def quat_scale_to_covar_preci_torch(quats, scales, compute_covar=True, compute_preci=True, triu=False):
    from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci
    return _quat_scale_to_covar_preci(quats, scales, compute_covar, compute_preci, triu)


@contextlib.contextmanager
def cpu_reference_kernels():
    saved = (gops.compute_relocation, gops.quat_scale_to_covar_preci)
    gops.compute_relocation = compute_relocation_torch
    gops.quat_scale_to_covar_preci = quat_scale_to_covar_preci_torch
    try:
        yield
    finally:
        gops.compute_relocation, gops.quat_scale_to_covar_preci = saved


# --------------------------------------------------------------------------------------------- state helpers
@torch.no_grad()
def zero_state_rows(optimizers, idx, names=None):
    """Zero every per-row optimiser state tensor (exp_avg, exp_avg_sq, row_step, ...) on rows idx."""
    for name, opt in optimizers.items():
        if names is not None and name not in names:
            continue
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p, {})
                for k, v in st.items():
                    if k == "step" or not isinstance(v, torch.Tensor) or v.dim() == 0:
                        continue
                    v[idx] = 0


@torch.no_grad()
def relocate_zeroing_dest(params, optimizers, mask, binoms, min_opacity):
    """gsplat relocate + zero the DESTINATION rows' optimiser state (gsplat zeroes only the source rows)."""
    dead = mask.nonzero(as_tuple=True)[0]
    gops.relocate(params=params, optimizers=optimizers, state={}, mask=mask, binoms=binoms, min_opacity=min_opacity)
    zero_state_rows(optimizers, dead)
    return int(len(dead))


class MCMCFixedBudget(MCMCStrategy):
    """gsplat MCMCStrategy with the destination-zeroing relocate and per-refine telemetry (n relocated / added)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.last = dict(step=-1, relocated=0, added=0)
        self.history = []

    def step_post_backward(self, params, optimizers, state, step, info, lr):
        state["binoms"] = state["binoms"].to(params["means"].device)
        binoms = state["binoms"]
        if step < self.refine_stop_iter and step > self.refine_start_iter and step % self.refine_every == 0:
            n0 = len(params["means"])
            n_rel = self._relocate_gs(params, optimizers, binoms)
            n_add = self._add_new_gs(params, optimizers, binoms)
            rec = dict(step=int(step), n_before=int(n0), relocated=int(n_rel), added=int(n_add),
                       n_after=int(len(params["means"])))
            self.last = rec
            self.history.append(rec)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gops.inject_noise_to_position(params=params, optimizers=optimizers, state={}, scaler=lr * self.noise_lr)

    @torch.no_grad()
    def _relocate_gs(self, params, optimizers, binoms):
        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = opacities <= self.min_opacity
        n_gs = int(dead_mask.sum().item())
        if n_gs > 0:
            relocate_zeroing_dest(params, optimizers, dead_mask, binoms, self.min_opacity)
        return n_gs


@torch.no_grad()
def reseed_rows(params, optimizers, xyz, rgb01, log_scale, max_n, min_opacity=0.005, init_opacity=0.1, sh0_c0=0.28209479177387814):
    """Coverage re-seeding: move dead rows first, then the lowest-opacity rows, onto the given RAFT points.
    xyz [M,3], rgb01 [M,3] in [0,1], log_scale [M] (isotropic). Returns the number of rows moved."""
    M = min(int(xyz.shape[0]), int(max_n))
    if M <= 0:
        return 0
    op = torch.sigmoid(params["opacities"].detach().flatten())
    order = torch.argsort(op, stable=True)                         # dead rows (lowest opacity) first
    idx = order[:M]
    dev = params["means"].device
    xyz, rgb01, log_scale = xyz[:M].to(dev), rgb01[:M].to(dev), log_scale[:M].to(dev)
    params["means"].data[idx] = xyz.to(params["means"].dtype)
    params["scales"].data[idx] = log_scale[:, None].expand(-1, 3).to(params["scales"].dtype)
    q = torch.zeros((M, 4), device=dev, dtype=params["quats"].dtype)
    q[:, 0] = 1.0
    params["quats"].data[idx] = q
    lo = torch.logit(torch.tensor(max(init_opacity, min_opacity * 2), device=dev))
    od = params["opacities"].data
    if od.dim() == 1:
        od[idx] = lo
    else:
        od[idx] = lo
    params["sh0"].data[idx] = ((rgb01 - 0.5) / sh0_c0)[:, None, :].to(params["sh0"].dtype)
    params["shN"].data[idx] = 0.0
    zero_state_rows(optimizers, idx)
    return int(M)
