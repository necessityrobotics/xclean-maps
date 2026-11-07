# xclean-maps

3D Gaussian-splat maps from [xclean](https://github.com/necessityrobotics/xclean) runs.

xclean-maps takes what an xclean user already has:

- an xclean output folder (`xclean replay`, `xclean live`, or `xclean run`), and
- the sequence folder that the run used,

and writes a standard 3D Gaussian-splat map (`splats.ply`) with its cameras, a held-out quality report and preview
images. It is the map builder that made the maps on the xclean website: the same training recipe, the same MCMC
densification, losses, exposure model, masks, sky rule and seeds (see CHANGES_FROM_INTERNAL.md for what we removed to
publish it).

Licence: Apache-2.0 (see LICENSE and NOTICE). Status: version 0.1.0, Windows 11 x64 + NVIDIA GPU (CUDA 13 driver).
Linux is untested.

## Quick start (Windows)

You need: an NVIDIA GPU with a recent driver (CUDA 13), Anaconda or Miniconda, and xclean's own environment.
No C++ compiler and no CUDA Toolkit are needed: we ship a prebuilt gsplat wheel.

**1. Make a map environment** (a copy of xclean's environment, so xclean itself stays as it is). In an Anaconda Prompt:

```
conda create -n xclean_maps --clone xclean
conda activate xclean_maps
```

No xclean environment yet? Make one first from xclean's `environment.yml` (see the xclean README), then clone it.

**2. Install** (about 3 GB, mostly PyTorch):

```
pip uninstall -y opencv-python
pip install opencv-contrib-python==5.0.0.93
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/necessityrobotics/xclean-maps/raw/v0.1.0/wheels/gsplat-1.4.0-cp312-cp312-win_amd64.whl
pip install https://github.com/necessityrobotics/xclean-maps/archive/refs/tags/v0.1.0.zip
```

Why the OpenCV swap: xclean pins `opencv-python` 5.0.0.93; OpenCV 5 keeps the HOG people detector (used for the
people mask) only in the contrib build. `opencv-contrib-python` is the same version with extra modules; xclean works
with it unchanged (`xclean doctor` checks it).

Check: `xclean-maps --version` and `xclean doctor`.

**3. Run xclean on your sequence** (any of these output folders works):

```
xclean run D:\data\my_route --out D:\runs\my_route
```

**4. Build the map** (`small` first: about 15 min on an RTX 5090 for a 2-minute route):

```
xclean-maps build D:\runs\my_route --out D:\maps\my_route --preset small
```

xclean-maps finds the sequence folder in the run's own files. If it cannot, it says so: then add
`--sequence D:\data\my_route`. When `small` looks right, build again with `--preset default` (the website recipe,
about 1 h on an RTX 5090) into a new `--out` folder.

**5. Look at it:**

```
xclean-maps view D:\maps\my_route --browser
```

This opens the map in your web browser (served from this PC only, at 127.0.0.1; Ctrl+C stops it). Mouse: left drag
orbits, right drag pans, the wheel zooms. Keys: W/A/S/D move, Q/E down/up, Space plays the camera path. "Camera view"
puts you in a recorded camera; the slider moves along the route.

A desktop window viewer is also there: `xclean-maps view D:\maps\my_route` (keys: W/S forward/back, A/D left/right,
Q/E down/up, drag to look, +/- speed, SPACE plays the camera path, TAB colour/depth, P screenshot, ESC quit).

Full install notes, the GPU list and troubleshooting: [INSTALL.md](INSTALL.md).

`splats.ply` is the standard INRIA 3DGS format, so other viewers (for example SuperSplat) open it too.

### Presets

| preset | training | splat budget | eval | use |
|---|---|---|---|---|
| `small` | max(10,000, 10 x training views) steps | 0.5 M to 1.0 M | standard | a quick check of an install or a sequence |
| `default` | max(30,000, 60 x training views) steps | max(2.0 M, 1.5 x 762 (grey) or 514 (colour) x training views), at most 6.0 M | standard | the recipe of the website maps |
| `full` | as `default` | as `default` | + held-out rotation alignment, off-trajectory views, transients | diagnosis |

Measured on our machine (RTX 5090), recipe `default`: a 1,005-view OAK-D-Lite route trained in 0.5 to 1.5 h
(28 to 84 ms per step); a 306-view EuRoC route in 11 min.

### What the build does

| stage | device | what |
|---|---|---|
| depth | GPU (CuPy) | xclean's 32-direction SGM on every keyframe (rectified grey pair, as the engine reads it) |
| prep | CPU | images, disparity, masks (valid, occlusion edges, saturation, sky, moving objects, people), frame quality, hold-out (every 10th keyframe), seeds, budget |
| train | GPU (gsplat) | one model for the whole route: MCMC with a fixed budget, visible-row Adam, L1 + SSIM + disparity Huber + coverage/sky BCE, per-image exposure, sky panorama, coverage re-seeding |
| eval | GPU | held-out PSNR / SSIM; no-GT render checks (coverage, PSNR dips, floaters, sky alpha), reported only |
| export | GPU + CPU | `splats.ply`, `cameras.json`, `pointcloud.ply`, `preview.png`, `previews/`, `keyframes/`, `report.json`, `report.md` |

Run the same command again after a stop: finished stages are skipped, and training continues from its last checkpoint
(every 30 min). `--fresh` starts training again. A file named `STOP` in `<out>/work/` stops training cleanly.

### The poses

- `xclean replay` / `xclean live` folders: the map uses the engine's own keyframes (`p1_frames.jsonl`) and the pose
  the engine published for each one (`recorder.jsonl`, `T_world_cam0_rectified`). The website maps used this path.
- `xclean run` folders: the map uses the keyframes and the final optimised keyframe poses of `kit/trajectory_kf.csv`.

The map frame is the run's world frame (metres, OpenCV camera axes).

### Output folder

```
<out>/splats.ply       the map (INRIA 3DGS PLY, SH degree 1)
<out>/cameras.json     every keyframe camera (position, camera-to-world rotation, fx, fy, cx, cy, T_world_cam, role)
<out>/report.json/.md  held-out PSNR / SSIM, checks, settings
<out>/preview.png      top view, held-out photo vs render, oblique view
<out>/previews/        held-out photo | render pairs
<out>/pointcloud.ply   the SGM32 depth as points
<out>/keyframes/       80 keyframe images + keyframes.json
<out>/work/            logs, plan, eval details, training curve, milestone checkpoints, live preview
<out>/cache/           depth and training arrays (large; `--clean-cache` removes it after the build)
```

## Limits

- The sky is not reconstructed as splats. The website maps show the splats over a plain gradient; the report gives the
  PSNR both on black and over that gradient.
- The no-GT render checks use the bars of our internal builder. They are reported, not enforced.
- xclean-maps applies no privacy blur. Check people, faces and number plates in your data before you share a map.
- GPU memory: the website maps were built on a 32 GB GPU. Smaller GPUs are untested; use `--preset small` first.

## Licence

Apache-2.0. Third-party notices (gsplat, GLM, xclean): NOTICE. A map is derived from your data: its licence follows
your data's licence.
