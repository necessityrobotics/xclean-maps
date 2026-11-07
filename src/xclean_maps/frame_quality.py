# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Frame-quality pre-pass (BUILDER 4.3). CPU. Bars are pre-registered in plan/THRESHOLDS.md before any training.

Per keyframe (on the RECTIFIED image the trainer uses):
  dark_share       share of pixels with grey <= 10/255
  sat_share        share of pixels with any channel >= 250/255
  mean_intensity   mean grey level (0-255)
  valid_share      share of pixels with RAFT disparity > 0.5 px that are neither saturated nor occlusion edges
                   (the pixels the depth loss and the seeds may use)
  lap_var          variance of the 3x3 Laplacian of the grey image (blur; OAK motorcycle vibration)
Rules (THRESHOLDS.md FQ-1..FQ-4):
  FQ-1 exclude from TRAINING if dark_share > 0.5 or sat_share > 0.5
  FQ-2 exclude from SEEDING if valid_share < 0.2
  FQ-3 exclude from TRAINING as blurred if lap_var < the row's 1st percentile AND both time-neighbour keyframes are
       sharp (lap_var >= the row's 10th percentile)
  FQ-4 excluded HELD-OUT frames stay in the evaluation; the headline is given with and without them
Saturated pixels are always dropped from the depth loss and the depth validity mask (masks.BIT_SAT).
"""
from __future__ import annotations

import numpy as np

from .masks import edge_mask, grey_of, saturated

FQ = dict(dark_level=10, sat_level=250, train_dark_max=0.5, train_sat_max=0.5, seed_valid_min=0.2,
          blur_pct=1.0, sharp_pct=10.0)


def measures(img, disp, scale=1.0, ok_mask=None):
    import cv2
    g = grey_of(img)
    sat = saturated(img, FQ["sat_level"])
    out = dict(dark_share=float((g <= FQ["dark_level"]).mean()), sat_share=float(sat.mean()),
               mean_intensity=float(g.mean()),
               lap_var=float(cv2.Laplacian(g, cv2.CV_64F, ksize=3).var()))
    if disp is None:
        out["valid_share"] = None
    else:
        # XCLEAN: ok_mask = the depth source's own valid & ~saturated & ~edge (masks.basic_bits), so SGM32's restated
        # valid / edge rules decide; None keeps the v4 RAFT expression
        ok = ok_mask if ok_mask is not None else ((disp > 0.5 * scale) & ~sat & ~edge_mask(disp, 1.0))
        out["valid_share"] = float(ok.mean())
    return out


def decide(rows, times=None, seed_valid_min=None):
    """rows: list of measure dicts in keyframe (time) order. -> list of dicts with exclusion flags + reasons."""
    lv = np.array([r["lap_var"] for r in rows], float)
    p1 = float(np.percentile(lv, FQ["blur_pct"])) if len(lv) else 0.0
    p10 = float(np.percentile(lv, FQ["sharp_pct"])) if len(lv) else 0.0
    out = []
    for i, r in enumerate(rows):
        reasons = []
        if r["dark_share"] > FQ["train_dark_max"]:
            reasons.append("dark_share %.3f > %.2f" % (r["dark_share"], FQ["train_dark_max"]))
        if r["sat_share"] > FQ["train_sat_max"]:
            reasons.append("sat_share %.3f > %.2f" % (r["sat_share"], FQ["train_sat_max"]))
        nb_sharp = (i > 0 and lv[i - 1] >= p10) and (i + 1 < len(lv) and lv[i + 1] >= p10)
        if lv[i] < p1 and nb_sharp:
            reasons.append("lap_var %.1f < p1 %.1f with both neighbours sharp (>= p10 %.1f)" % (lv[i], p1, p10))
        svm = FQ["seed_valid_min"] if seed_valid_min is None else seed_valid_min
        seed_ok = r["valid_share"] is not None and (r["valid_share"] > 0 if svm == 0 else r["valid_share"] >= svm)
        out.append(dict(exclude_train=bool(reasons), exclude_seed=(not seed_ok) or bool(reasons), reasons=reasons,
                        seed_reason=None if seed_ok else ("no depth" if r["valid_share"] is None else
                                                          "valid_share %.3f < %.2f" % (r["valid_share"], FQ["seed_valid_min"]))))
    return out, dict(lap_var_p1=p1, lap_var_p10=p10)
