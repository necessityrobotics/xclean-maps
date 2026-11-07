# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Recipe, camera families and presets of xclean-maps.

RECIPE and FAMILY are copied UNCHANGED from the internal builder that made the xclean website maps
(xclean_maps_v1 builder, config.py), except the operational key max_chunk_s (no chunked driver here; see
CHANGES_FROM_INTERNAL.md). No path is configured here: every path comes from the command line.

Presets (new in xclean-maps):
  default  the recipe of the website maps, unchanged: S = max(30,000, 60 x training views) steps, splat budget
           max(2.0 M, 1.5 x visible-per-view x training views) <= 6.0 M.
  full     the same training as default; eval also runs the diagnostic passes (held-out rotation alignment,
           every-keyframe coverage, off-trajectory views, transients).
  small    a quick test budget (NOT the website recipe): S = max(10,000, 10 x training views), splat budget
           0.5 M .. 1.0 M. Use it to check an install or a sequence, not for a final map.
"""
from __future__ import annotations

import copy

from . import util as U

RECIPE = dict(
    # views and hold-out (4.2)
    holdout_every=10, probe_views=64,
    # steps (5.4): S = max(30,000, 60 x n_train), epoch-shuffled, one view per step
    steps_per_photo=60, min_steps=30000,
    # densification (5.2): MCMC with a fixed budget
    refine_every=100, refine_stop_frac=0.80, min_opacity=0.005, noise_lr_normalised=5e5,
    # budgets (5.9) + lead decision: ceiling 6.0 M (no chunks/ partition anywhere, FULL viewable in a tab: A6, A7)
    budget_factor_old_map=1.5, visible_per_view={"rosario": 514, "rosariov2": 514, "oak": 762, "tartanair": 762, "euroc": 762,
                      "kitti": 762, "4seasons": 762, "madmax": 762},   # X-37: the grey families' value
    budget_floor=2_000_000, budget_ceiling=6_000_000,
    # learning rates (5.4)
    lr_means_per_L=1.6e-4, lr_means_final_frac=0.01, lr_scales=5e-3, lr_quats=1e-3, lr_opacities=5e-2,
    lr_sh0=2.5e-3, lr_shN=1.25e-4, adam_eps=1e-15, adam_betas=(0.9, 0.999),
    sh_degree=1, sh_ramp_steps=1000,
    # init (5.4)
    seed_every=4, seed_stride=8, seed_max_depth_m=60.0, seed_min_disp_px=2.0, seed_voxel_min_m=0.05,
    seed_voxel_frac=0.005, seed_cap_frac=0.5, init_opacity=0.5, init_scale_knn=0.1,
    # losses (5.5); INFERENCE starting weights
    w_l1=0.8, w_dssim=0.2, w_depth0=1.0, w_depth1=0.1, huber_delta_px=1.0, w_bce=0.05,
    w_opacity_reg=0.01, w_scale_reg=0.01, w_aniso=0.01, aniso_max_ratio=10.0, max_scale_m=10.0,
    max_angular_px=20.0, angular_cap_every=100,   # phase 2c (2026-10-22): sigma <= 20 px at the nearest training camera
                                                # (= the near-field p99 of the pilot; work/_diag_eval/footprint.json)
    w_exposure_prior=1e-2, clip_level=250.0 / 255.0, depth_alpha_min=0.05,
    w_near_depth=1.0,                  # phase 2a S3: one-sided penalty below the rig near clip (bounded, non-zero gradient)
    # exposure (5.6)
    exposure_on=True, exposure_start_frac=0.30, exposure_fit="closed form per step (phase 2a S2)", exposure_lam=1e-3,
    exposure_alpha_min=0.95, exposure_min_share=0.25, exposure_beta=0.25,
    # sky (5.7)
    sky_w=1024, sky_h=512, lr_sky=1e-2,                 # sky mask rule: masks.MASK_PARAMS (beyond 20 m)
    sky_photo_splat_grad=False,   # 2c: confident-sky pixels give the splats no photometric gradient (panorama only)
    sky6_maybe_as_sky=False,      # 2c arm cap20_sky6 (maybe-sky AND proxy trained as sky): NEGATIVE, K1 kill -1.4 dB probe; kept off
    # coverage passes and re-seeding (5.2, 7.1)
    coverage_every_frac=0.05, coverage_kf_stride=5, coverage_alpha=0.5, reseed_max_frac=0.02, reseed_stride_px=8,
    reseed_far_px=1.0,                 # 2c far-field arm: re-seeds beyond the seed near field (fB / seed_min_disp_px) get a
                                       # footprint of reseed_far_px pixels at their depth, clamped <= the near-field re-seed size
    coverage_full_at=(0.25, 0.50, 0.75, 1.00),
    # rasteriser (5.8)
    rasterize_mode="antialiased", eps2d=0.3, packed=False, near_plane=0.05, far_plane=1000.0,
    # memory (6)
    vram_fraction=0.80, vram_guard_gib=22.3, gpu_inputs_gib=18.0, bytes_per_gaussian=736,
    torch_threads=4, max_chunk_s=0,          # xclean-maps: one process trains to the end (internal: 1800 s chunks)
    # checkpoints / liveness (13)
    ckpt_every_s=1800, ckpt_keep=2, milestones=(0.25, 0.50, 0.75, 1.00), heartbeat_s=60,
    # live view (A5): N chosen so the measured overhead stays <= 3 % of step time
    live_overhead_max=0.03, live_every_init=250, live_every_min=50, live_every_max=5000, live_width=1600,
    # kill rules (11.4)
    k6_check_frac=0.10, k6_factor=1.5,
    # scale windows (5.4: the old planner's core limits; used ONLY for the local scale L, never for training)
    scale_window_max_kf=800, scale_window_max_path_m=400.0, scale_window_slack=1.15,
    # export (8)
    invisible_alpha=1.0 / 255.0, full_max_splats=8_000_000, full_max_bytes=900_000_000,
    pointcloud_max_points=2_000_000, keyframes_n=80,
    seed=0,
)

# per-family facts the builder needs beyond the kit (4.4, 4.5)
FAMILY = {
    "rosario": dict(channels=3, exposure="3+3", static_world_mask=False, camera_fixed_mask=False, people_mask=False,
                    rolling_shutter=True, ground_platform=True),
    "rosariov2": dict(channels=1, exposure="1+1", static_world_mask=False, camera_fixed_mask=False, people_mask=False,
                      rolling_shutter=False, ground_platform=True),
    "oak": dict(channels=1, exposure="1+1", static_world_mask=True, camera_fixed_mask=True, people_mask=True,
                rolling_shutter=False, ground_platform=True),
    "tartanair": dict(channels=3, exposure="3+3", static_world_mask=False, camera_fixed_mask=False, people_mask=False,
                      rolling_shutter=False, ground_platform=False),
    # XCLEAN: EuRoC MAV (grey, global shutter, a hand-carried / flying platform, INDOOR: no sky classes)
    "euroc": dict(channels=1, exposure="1+1", static_world_mask=False, camera_fixed_mask=False, people_mask=False,
                  rolling_shutter=False, ground_platform=False, sky=False),
    # X-37 (2026-10-30): the website's KITTI / 4Seasons / MADMAX routes, grey stereo as the board ran them, outdoor
    "kitti": dict(channels=1, exposure="1+1", static_world_mask=False, camera_fixed_mask=False, people_mask=False,
                  rolling_shutter=False, ground_platform=True, sky=True),
    "4seasons": dict(channels=1, exposure="1+1", static_world_mask=False, camera_fixed_mask=False, people_mask=False,
                  rolling_shutter=False, ground_platform=True, sky=True),
    "madmax": dict(channels=1, exposure="1+1", static_world_mask=False, camera_fixed_mask=False, people_mask=False,
                  rolling_shutter=False, ground_platform=True, sky=True),
}
for _f in FAMILY.values():                     # XCLEAN: outdoor families keep the v4 sky classes
    _f.setdefault("sky", True)

for _f in FAMILY.values():                     # outdoor families keep the sky classes (indoor families set sky=False)
    _f.setdefault("sky", True)

# privacy blur: no blur is applied (the internal builder had the same switch OFF); masks.privacy_blur() is a no-op
PRIVACY_BLUR_ENABLED = False
PRIVACY_NOTE = "no privacy blur is applied by xclean-maps; check people, plates and faces in your own data before sharing a map"

PRESETS = {
    "default": dict(),
    "full": dict(),
    "small": dict(min_steps=10000, steps_per_photo=10, budget_floor=500_000, budget_ceiling=1_000_000),
}
PRESET_EVAL_FULL = {"default": False, "full": True, "small": False}

# calib.json "source" (xclean docs/SEQUENCE_FORMAT.md) -> family; matched as a case-insensitive substring
FAMILY_FROM_SOURCE = (("euroc", "euroc"), ("kitti", "kitti"), ("4seasons", "4seasons"), ("madmax", "madmax"),
                      ("rosario v2", "rosariov2"), ("rosario", "rosario"), ("oak", "oak"), ("tartanair", "tartanair"))


# recipe keys that change nothing a model is trained FROM or computes (review N5d / SF4): kept out of every hash
OPERATIONAL_KEYS = ("ckpt_every_s", "ckpt_keep", "milestones", "heartbeat_s", "live_overhead_max", "live_every_init",
                    "live_every_min", "live_every_max", "live_width", "vram_guard_gib", "vram_fraction", "torch_threads",
                    "k1_enabled", "k6_check_frac", "k6_factor", "gpu_inputs_gib", "max_chunk_s")
# the recipe keys PREP reads (prep_row.py + scale.budget): only these enter the PREP / plan hash (SF4)
PREP_RECIPE_KEYS = ("bytes_per_gaussian", "holdout_every", "init_scale_knn", "probe_views", "scale_window_max_kf",
                    "scale_window_max_path_m", "scale_window_slack", "seed_cap_frac", "seed_every", "seed_max_depth_m",
                    "seed_min_disp_px", "seed_stride", "seed_voxel_frac", "seed_voxel_min_m", "camera_fixed_reviewed",
                    "budget_ceiling", "budget_factor_old_map", "budget_floor", "min_steps", "steps_per_photo",
                    "visible_per_view")


def recipe(preset="default", overrides=None):
    if preset not in PRESETS:
        raise SystemExit("unknown preset %r (known: %s)" % (preset, sorted(PRESETS)))
    r = copy.deepcopy(RECIPE)
    r.update(PRESETS[preset])
    r.update(overrides or {})
    return r


def config_hash(obj):
    return U.sha1_json(obj)


def prep_recipe(rec):
    return {k: rec.get(k) for k in PREP_RECIPE_KEYS}
