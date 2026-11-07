# Changes from the internal builder

Source: the internal map builder `xclean_maps_v1/builder` (about 10.9 k lines; it made the xclean website maps), read
only, on 2026-11-08. The sha1 of every source file used is in `SOURCE_SHA1.txt`.

Rule of this extraction: the maths and the defaults stay unchanged (training, MCMC strategy, losses, exposure, masks,
sky, seeds, frame quality, hold-out, budget, steps). Only internal ties are removed or made optional. Every change is
listed below.

## Files copied without a change in their maths

| package file | internal file | change |
|---|---|---|
| masks.py | masks.py | GSKY-1 path removed: `x33` / `x33g` sky rules now refuse (model not shipped). Privacy note text. Docstring path. |
| frame_quality.py, scale.py, losses.py, exposure.py, sky.py, render.py, mcmc_wrap.py, visible_adam.py, sampler.py, pose_refine.py, live.py, io_lib.py | same names | SPDX header only |
| evaluate.py | eval_row.py | renamed; SPDX header only |
| train.py | onego_train.py | (1) the quiet-flag stop is removed; the chunk limit stays, default off (`max_chunk_s` 0); (2) the global `STOP` file under the internal root is removed (the per-build `<out>/work/STOP` stays); (3) docstrings |
| config.py | config.py | RECIPE and FAMILY copied verbatim, except `max_chunk_s` 1800 -> 0 (operational key, outside every hash). Removed: every path, the row registry, licence texts of datasets, amendment tables, quiet flags. Added: PRESETS, FAMILY_FROM_SOURCE |
| util.py | util.py | write fence: the fixed internal roots -> `set_write_root(<--out>)`. Removed: BELOW_NORMAL priority, GPU-busy check, engine-process scan, nvidia-smi allow-list, GPU lock helpers, stage-marker helpers |

## Files rebuilt from internal code

| package file | internal source | what changed | why |
|---|---|---|---|
| runs.py (new) | sources.py `P1Run`, kit.py (not copied) | `ReplayRun` = `P1Run` unchanged in its reading rules (recorder.jsonl row-major T, LIVE-MAP / LIVE-ODOM choice, trajectory_live.csv fallback W LAST, keyframes = p1_frames.jsonl kf=true + first frame). New `KitRun` reads `xclean run` output (`kit/trajectory_kf.csv`, W LAST, final keyframe poses). New `no_gt_checks()`. `sequence_folder()` reads harness.json / feeder.json / kit/run_manifest.json | the frozen xsesto kit reader (kit.py, `xsesto.tools.build_map`) is an internal engine; a public user has only xclean's own outputs |
| sources.py | sources.py | kept: `rig_of`, `load_left_colour`, `uniq_margin`, `Sgm32Source`, `accel_up` (maths unchanged). Removed: frozen-engine import root (`XCLEAN_MAPS_ENGINE_ROOT`), row registry (`compactslam.bench.rows`), licence fence, RAFT A/B source. New: `open_dataset(folder)` with the installed xclean, `detect_ext` (xclean's png/jpg rule) | public users have the installed xclean only; RAFT was A/B-only and never in a shipped map |
| prep.py | prep_xclean.py + prep_row.py | `compute_depth` and `prepare` take a build spec + dirs instead of a row name. The SGM32 evidence arrays (`delta`, `uniq`) are written only with `--depth-evidence` or when a non-default sky rule (v2 / v3) needs them | the default sky rule v1 does not read them; they only fed a reported-only leak indicator (X-26.6). The masks, images, disparity and seeds are byte-identical (tested) |
| export.py | export_row.py | kept: `clean_params`, `pointcloud`, `keyframes`. `write_splats`: the dreamsim reader is replaced by our own reader + plyfile when installed (optional). `preview`: GT / ORB-SLAM3 traces removed; oblique size a parameter. New: `cameras_json`, `holdout_pairs`, `report_md` | dreamsim is internal; GT does not exist for users |
| pipeline.py (new) | build.py, stages_post.py, drive_xclean.py | the stages depth, prep, train, eval, export, with hashed markers and resume (as build.py). Removed: V0 pose gate, GPU_JOB.lock, quiet flags, BELOW_NORMAL child processes, arms, the overlap queue, kill-override files, run_config persistence of K2 / K6 / acceptance files, metadata / disclosure / page facts for the website, film and web stages | internal operations and website publishing |
| viewer.py (new) | the CPU splat script's `viewer()` (render_xclean_splat.py) | OpenCV fly-through; renders with gsplat on a GPU, or splat centres as points on a CPU | a simple local viewer |
| cli.py (new) | - | `xclean-maps build / view / info` | - |

## Dropped, with reasons

| internal item | decision | reason |
|---|---|---|
| V0 pose gate (gates/v0_*.py), GT scoring (`compactslam.bench`), GT alignment, ORB-SLAM3 traces | dropped | users have no ground truth. The no-GT run checks (pose validity, rotations, time order, coasted share, recorder end) are kept and reported in plan.json and report.json |
| render gates as a STOP before export (V2.1, V2.3, V3.1, SKY-6, P-a) | kept as REPORTED checks, never blocking | their bars were registered for the website rows; a user map should still export |
| V0 acceptance / waiver files, pilot labels, disclosure guard, page facts, amendments tables | dropped | website publishing process |
| GSKY-1 sky model (sky_prob.py, sky_model/) | not shipped | it failed its registered M bars (X-33, X-34); no website map used it (all built with sky rule v1). Its NOTICE text is prepared in NOTICE |
| RAFT A/B depth source, licence fence | dropped | A/B measurement only; nothing in the package imports a learned helper |
| films (films_xclean.py) | dropped | website asset; optional future command |
| web stage (node + splat-transform, LITE spz, site manifests) | optional `--web`, off by default, untested: calls `splat-transform` to write `map.sog` | website asset tooling with fixed internal paths |
| viewer_scores (staging-site backdrop port) | dropped | depends on the website's exported sky panorama; the gradient-backdrop score (X-32, the shipped headline) is kept |
| sky panorama file (sky_equirect.png) | not exported | owner decision X-32: the sky is not reconstructed in shipped maps |
| privacy R1 review sheet | dropped | internal review step; README tells users to check their data |
| rows_xclean.json, rows_v4.json, ab_*, chain_*.sh, review_*, gates/, audit/, _patches/, tests of internal rows | not shipped | internal registry and experiments |
| 30-min chunked driver, quiet flags | dropped; chunk limit kept as `--max-chunk-s` (default off) | one process trains to the end; checkpoints every 30 min and `--resume` (default) keep the resume path |
| process priority change (BELOW_NORMAL) | dropped | a user tool should not change its own priority without being asked |

## Operational values scaled to the GPU (outside every hash)

| key | internal | xclean-maps | reason |
|---|---|---|---|
| vram_guard_gib (K5) | 22.3 | min(22.3, 0.70 x GPU memory) | the internal value was set for a 32 GB GPU |
| gpu_inputs_gib | 18.0 | min(18.0, 0.55 x GPU memory) | same |
| max_chunk_s | 1800 | 0 (off) | no chunked driver |

On a 32 GB GPU the first two values are unchanged.

## Rules derived for inputs a public run may not have

| input | rule |
|---|---|
| keyframes of an `xclean replay` / `live` folder | `p1_frames.jsonl` kf=true + the first published frame (the internal rule; the release writes this file) |
| keyframes of an `xclean run` folder | every row of `kit/trajectory_kf.csv` |
| sequence folder | `--sequence`, else harness.json `sequence_folder`, else feeder.json `dataset`, else kit/run_manifest.json `dataset` |
| camera family | `--family`, else calib.json `source` (substring: euroc, kitti, 4seasons, madmax, rosario v2, rosario, oak, tartanair) |
| up vector | -g_w of the engine (IMU admitted) or of the final global BA (`xclean run`), else the accelerometer mean over the first 1 s (checked), else the mean camera -y (internal rule X-12) |

## Presets (new)

`default` = the internal recipe unchanged. `full` = `default` + the diagnostic eval passes. `small` = a test budget
(`min_steps` 10,000, `steps_per_photo` 10, `budget_floor` 0.5 M, `budget_ceiling` 1.0 M); NOT the website recipe.

## Verified identity (CPU test `TestPrepFewFrames.test_identity_with_internal_prep`)

On 8 keyframes of the OAK route 2026-03-27 16:40:02, with the internal builder's SGM32 depth rows: the package's PREP
gives byte-identical images, disparity and mask bits (except the moving-object bit, which depends on the neighbouring
views and so differs by construction with 8 views), and the same up vector, K and fB as the internal plan.
