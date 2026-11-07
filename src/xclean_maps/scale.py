# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Local scale L, gaussian budget and step count (BUILDER 5.4, 5.9; lead amendments A6/A7).

L = median over route windows of 1.1 x the maximum camera distance from the window mean. The windows are cut exactly as
the old planner cut chunk cores ([SL] plan_row: <= 800 KF and <= 400 m, slack 1.15). They exist ONLY to measure L; no
model is ever trained per window (A6: one jointly trained model per whole route).
"""
from __future__ import annotations

import math

import numpy as np


def path_cumlen(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def scale_windows(P, max_kf=800, max_m=400.0, slack=1.15):
    """[SL] plan_row core split: cores at equal quantiles of (index / n + path / total path)."""
    n = len(P)
    s = path_cumlen(P)
    n_win = max(1, math.ceil(n / max_kf), math.ceil(s[-1] / max_m))
    while True:
        c = np.arange(n) / max(n - 1, 1) + s / max(s[-1], 1e-9)
        edges = [0] + [int(np.searchsorted(c, 2.0 * k / n_win)) for k in range(1, n_win)] + [n]
        cores = [(edges[k], edges[k + 1]) for k in range(n_win)]
        ok = all((b - a) <= max_kf and (s[b - 1] - s[a]) <= max_m * slack for a, b in cores if b > a)
        if ok or n_win > 64:
            return cores
        n_win += 1


def local_scale(P, max_kf=800, max_m=400.0, slack=1.15):
    """L (metres) and the per-window values; the whole-route extent is never used."""
    cores = scale_windows(P, max_kf, max_m, slack)
    vals = []
    for a, b in cores:
        if b - a < 2:
            continue
        w = P[a:b]
        vals.append(1.1 * float(np.linalg.norm(w - w.mean(0), axis=1).max()))
    L = float(np.median(vals)) if vals else 1.0
    return max(L, 1.0), dict(windows=[[int(a), int(b)] for a, b in cores], window_scales=vals)


def budget(spec, n_train, recipe):
    """cap_max (5.9): rows with an old map 1.5 x its visible count; others 1.5 x the family's visible-per-view x n_train;
    floor 2.0 M; ceiling 6.0 M (lead: no chunks/ partition, FULL viewable in a tab)."""
    fam = spec["family"]
    old = spec.get("old_map") or {}
    if old.get("visible_count"):
        raw = recipe["budget_factor_old_map"] * float(old["visible_count"])
        rule = "1.5 x the old map's visible count (%d)" % old["visible_count"]
    else:
        vpv = recipe["visible_per_view"][fam]
        raw = recipe["budget_factor_old_map"] * vpv * n_train
        rule = "1.5 x %d visible per view (%s) x %d training views" % (vpv, fam, n_train)
    cap = int(min(max(raw, recipe["budget_floor"]), recipe["budget_ceiling"]))
    return cap, dict(raw=int(raw), rule=rule, floor=recipe["budget_floor"], ceiling=recipe["budget_ceiling"],
                     clamped=("floor" if raw < recipe["budget_floor"] else "ceiling" if raw > recipe["budget_ceiling"] else None))


def steps(n_train, recipe, steps_per_photo=None):
    spp = steps_per_photo or recipe["steps_per_photo"]
    return int(max(recipe["min_steps"], spp * n_train)), int(spp)


def model_gib(cap, bytes_per_gaussian=736):
    return cap * bytes_per_gaussian / 2 ** 30
