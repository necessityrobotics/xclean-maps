# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Loss terms (BUILDER 5.5). All data terms use valid pixels; regularisers use the VISIBLE rows only (5.3).

| term        | formula                                                                                  | weight          |
| photometric | 0.8 L1 + 0.2 (1 - SSIM) on clamp(a C + b) vs the photo, C = render over the sky panorama;  | 1               |
|             | clip-aware: photo >= 250/255 where the render is brighter is masked; moving / camera-fixed |                 |
|             | / privacy pixels masked                                                                    |                 |
| depth       | Huber(delta 1 px) of fB (1/ED - 1/z_RAFT) = fB/ED - d_RAFT on valid, non-sky, non-saturated,| 1.0 -> 0.1 log  |
|             | non-edge pixels with alpha > 0.05                                                          |                 |
| coverage/sky| BCE(alpha, target): 1 on valid RAFT pixels, 0 on sky pixels                                | 0.05            |
| opacity reg | mean(sigmoid(o[vis]))                                                                      | 0.01            |
| scale reg   | mean(exp(s[vis]) / L)                                                                      | 0.01            |
| anisotropy  | mean(max(max_s / min_s - 10, 0)) on vis                                                    | 0.01            |
| exposure    | lambda (|a - 1|^2 + |b|^2)                                                                 | 1e-2            |
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import io_lib as IO

# mask bits written by PREP (masks.py)
BIT_VALID = 1        # RAFT disparity > 0.5 px
BIT_EDGE = 2         # occlusion edge |grad d| > 1 px/px
BIT_SAT = 4          # any channel >= 250/255
BIT_SKY = 8          # stereo sky mask
BIT_MOVING = 16      # static-world test failed (moving object) or a detected person (OAK)
BIT_CAMFIXED = 32    # camera-fixed pixels (rig parts in frame)
BIT_PRIVACY = 64     # privacy-blurred pixels (switch OFF: owner decision; never set)
BIT_SKY_MAYBE = 128  # possible sky (phase 2a, SKY-2/3): excluded from depth loss, coverage positives and seeding, but
                     # NEVER a coverage target 0 (that is reserved for confident sky, BIT_SKY)


def bits(mask_u8, b):
    return (mask_u8 & b) != 0


def depth_weight(step, S, w0=1.0, w1=0.1):
    """1.0 -> 0.1, log-decay over S."""
    f = min(max(step / max(S, 1), 0.0), 1.0)
    return float(w0 * (w1 / w0) ** f)


def photometric(pred, gt, mask, win, clip_level=250.0 / 255.0, w_l1=0.8, w_dssim=0.2):
    """pred, gt [H, W, 3] in [0, 1]; mask [H, W] bool (pixels allowed in the loss). Returns (loss, l1, ssim)."""
    clipped = (gt >= clip_level) & (pred > gt)                         # per channel: brighter than a clipped photo
    m3 = mask[..., None] & ~clipped
    m3f = m3.to(pred.dtype)
    l1 = (torch.abs(pred - gt) * m3f).sum() / m3f.sum().clamp_min(1.0)
    pix = mask & ~clipped.any(-1)
    s = IO.ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], win, mask=pix)
    return w_l1 * l1 + w_dssim * (1.0 - s), l1, s


def huber(r, delta):
    a = torch.abs(r)
    return torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def disparity_loss(ed, alpha, disp, mask, fB, delta=1.0, alpha_min=0.05, z_min=1e-3, w_near=1.0):
    """ed [H, W] expected depth, alpha [H, W], disp [H, W] RAFT disparity (px), mask [H, W] bool (valid, non-sky,
    non-saturated, non-edge). Returns the mean loss in DISPARITY pixels over the used pixels.
    The Huber term clamps ED at z_min (the rig's near clip), because d(fB/ED)/dED = -fB/ED^2 is unbounded as ED -> 0.
    Phase 2a (review S3): below the clamp a ONE-SIDED bounded penalty w_near * delta * (z_min - ED) / z_min keeps a
    non-zero gradient that pushes too-near renders (near-camera floaters) back out, instead of no gradient at all."""
    use = mask & (alpha > alpha_min)
    if not bool(use.any()):
        return ed.sum() * 0.0, 0
    e = ed[use]
    r = fB / torch.clamp(e, min=z_min) - disp[use]
    near = torch.relu(z_min - e) / z_min
    return (huber(r, delta) + w_near * delta * near).mean(), int(use.sum())


def coverage_targets(b):
    """(pos, neg) of the coverage / sky BCE (review M1 / SKY-1): positives = valid RAFT pixels that are not moving,
    saturated, SKY or SKY_MAYBE; negatives = confident SKY only. The two sets are disjoint by construction."""
    pos = ((b & BIT_VALID) != 0) & ((b & (BIT_MOVING | BIT_SAT | BIT_SKY | BIT_SKY_MAYBE)) == 0)
    neg = (b & BIT_SKY) != 0
    return pos, neg


def coverage_bce(alpha, pos, neg):
    """BCE(alpha, 1) on pos pixels (valid RAFT), BCE(alpha, 0) on neg pixels (sky)."""
    a = alpha.clamp(1e-6, 1 - 1e-6)
    n = pos.sum() + neg.sum()
    if int(n) == 0:
        return alpha.sum() * 0.0
    lp = -(torch.log(a) * pos.to(a.dtype)).sum()
    ln = -(torch.log(1 - a) * neg.to(a.dtype)).sum()
    return (lp + ln) / n.to(a.dtype)


def regularisers(params, vis, L, w_op=0.01, w_scale=0.01, w_aniso=0.01, max_ratio=10.0):
    """Regularisers on the VISIBLE rows only; returns (total, parts). Zero (and no graph) when no row is visible."""
    idx = torch.nonzero(vis, as_tuple=True)[0]
    if idx.numel() == 0:
        z = params["means"].sum() * 0.0
        return z, dict(opacity=0.0, scale=0.0, aniso=0.0)
    o = torch.sigmoid(params["opacities"].reshape(-1)[idx])
    s = torch.exp(params["scales"][idx])
    r_op = o.mean()
    r_sc = s.mean() / float(L)
    ratio = s.max(1).values / s.min(1).values.clamp_min(1e-12)
    r_an = F.relu(ratio - max_ratio).mean()
    tot = w_op * r_op + w_scale * r_sc + w_aniso * r_an
    return tot, dict(opacity=float(r_op.detach()), scale=float(r_sc.detach()), aniso=float(r_an.detach()))


def nearest_cam_dist(means, cams, chunk=65536):
    """Distance (m) from every row's centre to the nearest camera centre (cams: (C, 3)), chunked cdist."""
    out = torch.empty(len(means), dtype=means.dtype, device=means.device)
    for i in range(0, len(means), chunk):
        out[i:i + chunk] = torch.cdist(means[i:i + chunk], cams).min(1).values
    return out


def angular_cap_log(dist, fx, max_px, floor_m=0.05):
    """Log of the largest sigma (m) whose angular size at distance `dist` is max_px pixels: log(max_px x dist / fx)."""
    return torch.log(max_px * dist.clamp_min(floor_m) / float(fx))


def clamp_scales_angular_(params, cap_log):
    """Distance-relative size clamp (phase 2c, 2026-10-22): log-scale <= cap_log per row (a (N,) tensor)."""
    with torch.no_grad():
        s = params["scales"].data
        torch.minimum(s, cap_log[:, None], out=s)


def clamp_scales_(params, max_scale_m=10.0):
    """Hard size clamp (5.5): log-scale <= log(10 m) after each step."""
    with torch.no_grad():
        params["scales"].data.clamp_(max=math.log(max_scale_m))
