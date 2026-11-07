# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Per-image affine exposure (BUILDER 5.6): pred = clamp(a * C + b, 0, 1) per training photo.

Colour rows (Rosario v1, TartanAir): a gain and a bias per channel (3 + 3). Grey rows (OAK, Rosario v2 IR): 1 + 1.
Held at identity until exposure_start_frac x S (0.3 S since phase 2a; geometry forms first). Prior lambda (|a-1|^2 + |b|^2).

Phase 2a (review S2): the phase-1 gradient fit (lr 1e-3 -> 1e-4, each photo updated once per epoch) could move a gain by
only about 2 % over a whole run. The exposure is now fitted IN CLOSED FORM for the current photo at every step, against
the current render (detached), with the same prior:
    min_{a,b} mean_p (a c_p + b - i_p)^2 + lambda ((a - 1)^2 + b^2)        (per channel; grey: one channel)
over unsaturated, unmasked pixels. lambda = recipe exposure_lam = 1e-3 (per-pixel-mean units): the shrinkage towards
identity is lambda / (var(c) + lambda) of the deviation, <= 3 % at an image variance >= 0.03 (a +30 % gain is reached
to within 0.01; tests/test_phase2a); the phase-1 weight 1e-2 would have shrunk a +30 % gain to about +24 %. The solution is stored per photo (checkpointed) and used for the loss; held-out photos
use (a, b) interpolated in time from the two time-adjacent training photos. The exported map is the canonical
(identity-exposure) appearance; the headline PSNR is raw (identity).

GAUGE (phase 2a): a per-photo affine exposure and a shared map have a global ambiguity (map x k with every gain / k).
A per-step closed-form fit lets the map drift dark or bright while every photo's gain compensates. The exposure applied in
the loss is therefore NORMALISED over the photos fitted so far: pred = ((a_i C + b_i) - b_mean) / a_gmean (per channel;
a_gmean = geometric mean of the fitted gains, b_mean = mean fitted bias), so the exposure common to all photos is the
identity by construction and the map carries the average appearance; only per-photo deviations are absorbed.
The trainer fits only on reliable pixels (render alpha >= 0.95, not sky / maybe sky, unsaturated, >= 25 % of the frame)
and smooths each photo's estimate over its visits (beta); see THRESHOLDS.md 6.7 for the toy A/B and the open decision.
"""
from __future__ import annotations

import numpy as np
import torch


class Exposure:
    def __init__(self, n_train, channels, device="cpu"):
        self.ce = int(channels)
        init = torch.zeros((int(n_train), 2 * self.ce), dtype=torch.float32, device=device)
        init[:, :self.ce] = 1.0
        self.param = torch.nn.Parameter(init, requires_grad=False)

    def a_b(self, i):
        row = self.param[i]
        return row[:self.ce], row[self.ce:]

    def gauge(self):
        """(a_gmean, b_mean) over the rows fitted so far (a row exactly at identity has never been fitted)."""
        P = self.param.detach()
        a, b = P[:, :self.ce], P[:, self.ce:]
        fitted = ~((a == 1.0).all(1) & (b == 0.0).all(1))
        if not bool(fitted.any()):
            return torch.ones(self.ce, device=P.device), torch.zeros(self.ce, device=P.device)
        return torch.exp(torch.log(a[fitted].clamp_min(1e-3)).mean(0)), b[fitted].mean(0)

    def normalised(self, i):
        """(a_i / a_gmean, (b_i - b_mean) / a_gmean): the exposure the loss applies (see GAUGE)."""
        a, b = self.a_b(i)
        ga, gb = self.gauge()
        return a / ga, (b - gb) / ga

    def apply(self, img_hw3, i):
        a, b = self.normalised(i)
        if self.ce == 1:
            a, b = a.expand(3), b.expand(3)
        return torch.clamp(img_hw3 * a + b, 0.0, 1.0)

    @torch.no_grad()
    def solve_gain(self, C, img, mask, i, beta=1.0, min_px=64, lo=0.05, hi=0.95):
        """ROBUST global gain (one value for every channel, bias 0): the median of photo / render over well-exposed
        pixels (render and photo in [lo, hi]); local render errors move a median far less than a least-squares fit."""
        m = mask & (C.min(-1).values >= lo) & (C.max(-1).values <= hi) & (img.max(-1).values <= hi) & (img.min(-1).values >= lo)
        if int(m.sum()) < max(64, int(min_px)):
            return self.a_b(i)
        g = float(torch.median(img[m].mean(-1) / C[m].mean(-1).clamp_min(1e-3)))
        new = torch.cat([torch.full((self.ce,), g), torch.zeros(self.ce)]).to(self.param)
        old = self.param[i].clone()
        fitted_before = not (bool((old[:self.ce] == 1.0).all()) and bool((old[self.ce:] == 0.0).all()))
        self.param[i] = old + float(beta) * (new - old) if fitted_before else new
        return self.a_b(i)

    @torch.no_grad()
    def solve(self, C, img, mask, i, lam=1e-3, beta=1.0, min_px=64):
        """Closed-form affine fit of the render C [H, W, 3] to the photo img [H, W, 3] on mask [H, W]; stores row i as
        old + beta (fit - old) once the row has been fitted before (beta = 1: the fit itself; the first fit is stored as is)."""
        m = mask.reshape(-1)
        if int(m.sum()) < max(64, int(min_px)):
            return self.a_b(i)
        X = C.reshape(-1, 3)[m].double()
        Y = img.reshape(-1, 3)[m].double()
        if self.ce == 1:
            X, Y = X.mean(1, keepdim=True), Y.mean(1, keepdim=True)
        n = float(X.shape[0])
        sxx, sx, sxy, sy = (X * X).sum(0) / n, X.sum(0) / n, (X * Y).sum(0) / n, Y.sum(0) / n
        a11, a12, a22 = sxx + lam, sx, 1.0 + lam
        r1, r2 = sxy + lam, sy
        det = a11 * a22 - a12 * a12
        a = (r1 * a22 - a12 * r2) / det
        b = (a11 * r2 - a12 * r1) / det
        new = torch.cat([a, b]).float()
        old = self.param[i].clone()
        fitted_before = not (bool((old[:self.ce] == 1.0).all()) and bool((old[self.ce:] == 0.0).all()))
        self.param[i] = old + float(beta) * (new - old) if fitted_before else new
        return self.a_b(i)

    def prior(self, i):
        a, b = self.a_b(i)
        return ((a - 1.0) ** 2).sum() + (b ** 2).sum()

    def is_identity(self, i):
        a, b = self.a_b(i)
        return bool(torch.all(a == 1.0) and torch.all(b == 0.0))

    def interpolated(self, t_query, t_train):
        """(a, b) for a held-out photo at time t_query from the time-adjacent training photos (numpy, detached), in the
        same normalised gauge as apply()."""
        ga, gb = self.gauge()
        P = self.param.detach().clone()
        P[:, :self.ce] = P[:, :self.ce] / ga
        P[:, self.ce:] = (P[:, self.ce:] - gb) / ga
        P = P.cpu().numpy()
        t_train = np.asarray(t_train, float)
        o = np.argsort(t_train)
        tt, PP = t_train[o], P[o]
        j = int(np.searchsorted(tt, t_query))
        if j <= 0:
            row = PP[0]
        elif j >= len(tt):
            row = PP[-1]
        else:
            w = (t_query - tt[j - 1]) / max(tt[j] - tt[j - 1], 1e-9)
            row = (1 - w) * PP[j - 1] + w * PP[j]
        return row[:self.ce].copy(), row[self.ce:].copy()


def apply_numpy(img, a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    if a.size == 1:
        a, b = np.repeat(a, 3), np.repeat(b, 3)
    return np.clip(img * a[None, None] + b[None, None], 0.0, 1.0)
