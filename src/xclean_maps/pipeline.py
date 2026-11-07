# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""The build: depth -> prep -> train -> eval -> export, for ONE map, everything under --out.

  <out>/work/    build_spec.json, plan.json, logs, markers, held-out eval, training curve, milestone checkpoints
  <out>/cache/   depth (SGM32), the training arrays (img / disp / mask .npy), seeds, views, rolling checkpoints
  <out>/         the map: splats.ply, cameras.json, report.json, report.md, preview.png, previews/, pointcloud.ply,
                 keyframes/ (see export.py)

Every stage writes a marker <out>/work/markers/.done_<stage> with the hash of its inputs. A stage whose marker holds the
current hash is skipped; otherwise its outputs are replaced. So `xclean-maps build ...` run again after a stop continues
where it stopped; training resumes from its newest rolling checkpoint (every 30 min) when --resume is given (default on).
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import config as CFG
from . import util as U

STAGES = ("depth", "prep", "train", "eval", "export")
STAGE_VERSION = dict(depth="depth-1", prep="prep-1", train="train-1", eval="eval-1", export="export-1")


def make_dirs(out):
    out = os.path.abspath(out).replace("\\", "/")
    return dict(out=out, work=out + "/work", hot=out + "/cache")


def family_from_calib(seq):
    """The camera family from calib.json "source" (and the image channels); None when unknown."""
    cal = U.read_json(os.path.join(seq, "calib.json"), {}) or {}
    src = str(cal.get("source") or "").lower()
    for key, fam in CFG.FAMILY_FROM_SOURCE:
        if key in src:
            return fam
    return None


def resolve_spec(run_dir, sequence=None, out=None, family="auto", preset="default", limit=None, steps=None, ext=None,
                 name=None, gate=True, depth_evidence=False):
    from . import runs as RUNS
    run_dir = os.path.abspath(run_dir).replace("\\", "/")
    run = RUNS.open_run(run_dir)
    seq = sequence or run.sequence_folder()
    if not seq:
        raise SystemExit("the run folder does not record its sequence folder: give --sequence <folder>")
    seq = os.path.abspath(seq).replace("\\", "/")
    if not os.path.isfile(os.path.join(seq, "calib.json")):
        raise SystemExit("%s is not a sequence folder (no calib.json); give --sequence <folder>" % seq)
    fam = family
    if fam in (None, "auto"):
        fam = family_from_calib(seq)
        if fam is None:
            raise SystemExit("cannot tell the camera family from %s/calib.json 'source'; give --family (one of %s)"
                             % (seq, sorted(CFG.FAMILY)))
    if fam not in CFG.FAMILY:
        raise SystemExit("unknown family %r (known: %s)" % (fam, sorted(CFG.FAMILY)))
    if preset not in CFG.PRESETS:
        raise SystemExit("unknown preset %r (known: %s)" % (preset, sorted(CFG.PRESETS)))
    return dict(name=name or os.path.basename(os.path.abspath(out or run_dir)), run_dir=run_dir, sequence=seq,
                family=fam, preset=preset, limit=limit, steps_override=steps, ext=ext, gate=bool(gate),
                train_scale=1.0, depth_source="sgm32", mask_overrides=None, depth_evidence=bool(depth_evidence),
                label="xclean-maps build", old_map=None)


# --------------------------------------------------------------------------------------------- markers
def _marker(dirs, st):
    return os.path.join(dirs["work"], "markers", ".done_%s" % st)


def marker_ok(dirs, st, h):
    m = U.read_json(_marker(dirs, st))
    return bool(m and m.get("hash") == h and m.get("ok"))


def mark(dirs, st, h, ok, info=None):
    U.ensure_dir(os.path.join(dirs["work"], "markers"))
    U.atomic_json(_marker(dirs, st), dict(stage=st, hash=h, ok=bool(ok), finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          info=info or {}))


def clear_marker(dirs, st):
    U.remove_if_exists(_marker(dirs, st))


def stage_hash(spec, dirs, st, rec):
    from . import masks as MK
    base = dict(stage=st, version=STAGE_VERSION[st])
    if st == "depth":
        from . import runs as RUNS
        run = RUNS.open_run(spec["run_dir"])
        base.update(run=run.dir, pose_file=(run.pose_file or {}).get("sha1"), sequence=spec["sequence"],
                    calib=U.sha1_file(os.path.join(spec["sequence"], "calib.json")), limit=spec.get("limit"),
                    evidence=bool(spec.get("depth_evidence")), sky_rule=MK.row_mask_params(spec).get("sgm32_sky_rule"))
    elif st == "prep":
        from . import frame_quality as FQ
        di = os.path.join(dirs["hot"], "depth", "depth_index.json")
        base.update(depth_index=U.sha1_file(di) if os.path.isfile(di) else None, recipe=CFG.prep_recipe(rec),
                    mask=MK.row_mask_params(spec), fq=FQ.FQ, family=spec["family"], steps=spec.get("steps_override"),
                    limit=spec.get("limit"))
    elif st == "train":
        plan = U.read_json(dirs["work"] + "/plan.json") or {}
        base.update(plan=plan.get("config_hash"), recipe={k: v for k, v in rec.items() if k not in CFG.OPERATIONAL_KEYS})
    elif st == "eval":
        base.update(final=(U.read_json(dirs["work"] + "/train_summary.json", {}) or {}).get("final_params_sha1"),
                    full=CFG.PRESET_EVAL_FULL[spec["preset"]])
    elif st == "export":
        ev = dirs["work"] + "/holdout_eval.json"
        base.update(final=(U.read_json(dirs["work"] + "/train_summary.json", {}) or {}).get("final_params_sha1"),
                    eval=U.sha1_file(ev) if os.path.isfile(ev) else None)
    return U.sha1_json(base)


# --------------------------------------------------------------------------------------------- devices
def setup_device(device, rec, log):
    """-> 'cuda' or 'cpu'. On CUDA: the per-process memory fraction (recipe vram_fraction 0.80) and the operational
    memory keys scaled to the GPU (the website maps ran on a 32 GB GPU: guard 22.3 GiB, resident inputs <= 18 GiB)."""
    import torch
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda, but PyTorch sees no CUDA GPU (see INSTALL.md)")
        torch.cuda.set_per_process_memory_fraction(rec["vram_fraction"], 0)
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        rec["vram_guard_gib"] = float(min(rec["vram_guard_gib"], 0.70 * tot))
        rec["gpu_inputs_gib"] = float(min(rec["gpu_inputs_gib"], 0.55 * tot))
        log("GPU %s, %.1f GiB; memory fraction %.2f; K5 guard %.1f GiB; resident inputs <= %.1f GiB"
            % (torch.cuda.get_device_name(0), tot, rec["vram_fraction"], rec["vram_guard_gib"], rec["gpu_inputs_gib"]))
        return "cuda"
    log("device CPU: the reference rasteriser is used; this is for tiny tests only (a real map needs a CUDA GPU)")
    return "cpu"


def renderer(dev, rec):
    from .render import gsplat_renderer, reference_renderer
    if dev == "cuda":
        return gsplat_renderer(rec["rasterize_mode"], rec["packed"], rec["near_plane"], rec["far_plane"], rec["eps2d"])
    return reference_renderer(antialiased=rec["rasterize_mode"] == "antialiased", near_plane=rec["near_plane"],
                              far_plane=rec["far_plane"], eps2d=rec["eps2d"])


def load_model(dirs, dev, rec, name=None, from_ply=None):
    """(prep, Model) from the trained checkpoint (sky panorama + exposure) or from a splats.ply."""
    import torch
    from . import exposure as EX
    from . import sky as SKY
    from .evaluate import Model, params_to_torch
    from .io_lib import params_from_splat_ply
    from .prep import Prepared
    prep = Prepared(dirs, name)
    rend = renderer(dev, rec)
    if from_ply:
        return prep, Model(params_to_torch(params_from_splat_ply(from_ply), dev), rend, dev, label="splats.ply")
    blob = torch.load(dirs["work"] + "/final_params.pt", map_location="cpu", weights_only=False)
    params = {k: v.to(dev) for k, v in blob["params"].items()}
    ch = 1 if prep.channels == 1 else 3
    sky = SKY.SkyPanorama(prep.up, ch, rec["sky_w"], rec["sky_h"], device=dev)
    sky.param = torch.nn.Parameter(blob["sky"].to(dev))
    sky.min_lat_deg = rec.get("sky_pano_min_lat_deg")
    ex = EX.Exposure(len(prep.train_ids), ch, device=dev)
    ex.param = torch.nn.Parameter(blob["expo"].to(dev), requires_grad=False)
    t_train = [prep.views[j]["t"] for j in prep.train_ids]
    return prep, Model(params, rend, dev, sky=sky, expo=ex, t_train=t_train)


# --------------------------------------------------------------------------------------------- stages
def stage_depth(spec, dirs, rec, a, log):
    from . import prep as PX
    if a.device != "cuda":
        raise SystemExit("the depth stage (32-direction SGM) runs on the GPU with CuPy: --device cuda is needed")
    import cupy as cp
    cp.cuda.Device(0).use()
    tot = cp.cuda.Device(0).mem_info[1]
    cp.get_default_memory_pool().set_limit(size=int(0.85 * tot))      # GPU memory share <= 0.85
    PX.compute_depth(spec, dirs, log=log, limit=spec.get("limit"))
    return True


def stage_prep(spec, dirs, rec, a, log):
    from . import prep as PX
    plan = PX.prepare(spec, dirs, rec, workers=a.workers, log=log, limit=spec.get("limit"))
    if not plan.get("prep_pass", True):
        log("PREP sky scan FAILED: %s (see work/sky_scan.json): REPORTED, the map is still trained" % plan.get("sky_scan"))
    return True


def stage_train(spec, dirs, rec, a, log):
    import torch
    from .train import DataSource, Trainer
    from .prep import Prepared
    from . import scale as SC
    prep = Prepared(dirs, spec["name"])
    for st in ("eval", "export"):
        clear_marker(dirs, st)
    torch.set_num_threads(rec["torch_threads"])
    U.record_pid(os.path.join(dirs["work"], "pids.json"), "train")
    model = SC.model_gib(prep.plan["cap_max"])
    resident = a.device == "cuda" and 1.4 * model + prep.plan["inputs_gb"] / 1.073741824 <= rec["gpu_inputs_gib"]
    log("inputs %.2f GB, model %.2f GiB -> inputs %s; exposure %s; preset %s; steps %d; splat budget %d" % (
        prep.plan["inputs_gb"], model, "on the GPU" if resident else "in host RAM", "ON" if rec.get("exposure_on", True)
        else "OFF", spec["preset"], prep.plan["steps"], prep.plan["cap_max"]))
    data = DataSource(prep, a.device, resident)
    run_config = dict(ack_kill=a.ack_kill) if getattr(a, "ack_kill", None) else {}
    tr = Trainer(prep, rec, a.device, renderer(a.device, rec), dirs["work"], dirs["hot"], run_config=run_config,
                 live=not a.no_live)
    info = tr.run(data, resume=a.resume, stop_after=a.stop_after)
    return bool(info) and not info.get("stopped")


def stage_eval(spec, dirs, rec, a, log):
    from . import evaluate as EV
    prep, model = load_model(dirs, a.device, rec, spec["name"])
    w = dirs["work"]
    for fn in ("holdout_eval.json", "every_kf.json", "render_checks.json", "gradient_scores.json"):
        U.remove_if_exists(w + "/" + fn)
    fsha = (U.read_json(w + "/train_summary.json", {}) or {}).get("final_params_sha1")
    full = CFG.PRESET_EVAL_FULL[spec["preset"]]
    h = EV.eval_heldout(model, prep, align=full)
    h["final_params_sha1"] = fsha
    U.atomic_json(w + "/holdout_eval.json", h)
    k = EV.eval_every_kf(model, prep)
    k["final_params_sha1"] = fsha
    U.atomic_json(w + "/every_kf.json", k)
    gs = EV.gradient_scores(EV.Model(model.params, model.render, a.device, label="gradient"), prep)
    U.atomic_json(w + "/gradient_scores.json", gs)
    checks = dict(v21_coverage=k["v21"], v23_psnr_dips=k["v23"], v31_floaters=h["v31"], sky6_sky_alpha=h["sky6"],
                  p_a_psnr_floor=h["p_a"])
    if full:
        checks.update(v36_off_trajectory=EV.eval_off_trajectory(model, prep), v37_transients=EV.eval_transients(model, prep))
    U.atomic_json(w + "/render_checks.json", dict(checks, final_params_sha1=fsha, note=(
        "no-GT render checks of the internal builder (bars inherited from its THRESHOLDS); REPORTED, never blocking")))
    bad = sorted(n for n, v in checks.items() if isinstance(v, dict) and "pass_" in v and not v["pass_"])
    log("eval: %d held-out views; PSNR %.2f dB splats on black, %.2f dB over the gradient backdrop, %.2f dB with the "
        "trainer's sky panorama; SSIM %.4f; no-GT checks failing (reported only): %s"
        % (h["all"]["n"], h["all"]["psnr_raw_nosky"], gs["all"]["psnr_db"], h["all"]["psnr_raw"], h["all"]["ssim"],
           bad or "none"))
    return True


def stage_export(spec, dirs, rec, a, log):
    from . import evaluate as EV
    from . import export as EXR
    md = dirs["out"]
    w = dirs["work"]
    prep, model = load_model(dirs, a.device, rec, spec["name"])
    outs = ("splats.ply", "cameras.json", "pointcloud.ply", "preview.png", "report.json", "report.md")
    for fn in outs + ("keyframes", "previews", "map.sog"):
        U.remove_if_exists(os.path.join(md, fn))
    np_params = {k: v.detach().cpu().numpy() for k, v in model.params.items()}
    clean, cinfo = EXR.clean_params(np_params, prep.channels)
    sinfo = EXR.write_splats(md + "/splats.ply", clean)
    _, mfile = load_model(dirs, a.device, rec, spec["name"], from_ply=md + "/splats.ply")
    ea = EV.eval_heldout(EV.Model(model.params, model.render, a.device, label="trained, no sky"), prep, align=False)
    eb = EV.eval_heldout(mfile, prep, align=False)
    d = abs(ea["all"]["psnr_raw_nosky"] - eb["all"]["psnr_raw_nosky"])
    rescore = dict(trained_nosky_db=ea["all"]["psnr_raw_nosky"], file_nosky_db=eb["all"]["psnr_raw_nosky"], abs_diff_db=d,
                   pass_=bool(d <= 0.02))
    cinf = EXR.cameras_json(md + "/cameras.json", prep)
    pinfo = EXR.pointcloud(md + "/pointcloud.ply", prep, prep.plan["near_clip_m"], prep.plan["max_depth_m"])
    kinfo = EXR.keyframes(md + "/keyframes", prep)
    vinfo = EXR.preview(md + "/preview.png", prep, model, md + "/pointcloud.ply", getattr(a, "oblique_size", (1280, 720)))
    pairs = EXR.holdout_pairs(md + "/previews", prep, model)
    hold = U.read_json(w + "/holdout_eval.json", {}) or {}
    gs = U.read_json(w + "/gradient_scores.json", {}) or {}
    rc = U.read_json(w + "/render_checks.json", {}) or {}
    ts = U.read_json(w + "/train_summary.json", {}) or {}
    plan = prep.plan
    al = hold.get("all") or {}
    from . import __version__
    rep = dict(
        name=spec["name"], xclean_maps_version=__version__, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
        preset=spec["preset"], family=spec["family"], run_dir=spec["run_dir"], sequence=spec["sequence"],
        pose_stream=plan.get("pose_stream"), layout=(plan.get("run_checks") or {}).get("layout"),
        heldout=dict(n=al.get("n"), psnr_splats_on_black_db=al.get("psnr_raw_nosky"),
                     psnr_gradient_backdrop_db=(gs.get("all") or {}).get("psnr_db"),
                     psnr_with_sky_panorama_db=al.get("psnr_raw"), psnr_exposure_interpolated_db=al.get("psnr_expo"),
                     ssim=al.get("ssim"), without_excluded=hold.get("without_excluded"),
                     per_stretch=hold.get("per_stretch"),
                     rule=("held-out = every 10th keyframe, never trained on; identity exposure; at the run's own poses; "
                           "'splats on black' = what a plain 3DGS viewer shows")),
        rescore=rescore, n_splats=sinfo["n"], n_trained=cinfo["n_trained"], n_invisible_removed=cinfo["n_invisible_removed"],
        splats_bytes=sinfo["bytes"], splats_sha1=sinfo["sha1"], ply_schema=sinfo["schema"], cameras=cinf, pointcloud=pinfo,
        keyframes=kinfo, preview=vinfo, previews=pairs, steps=ts.get("steps"), train_seconds=ts.get("wall_s"),
        ms_per_step=ts.get("ms_per_step"), n_views=plan.get("n_views"), n_train=plan.get("n_train"), n_hold=plan.get("n_hold"),
        image_size=plan.get("image_size"), K_rect=plan.get("K_rect"), fB=plan.get("fB"), up=plan.get("up"),
        up_rule=plan.get("up_rule"), render_checks={k: v for k, v in rc.items() if isinstance(v, dict)},
        sky_scan_pass=(plan.get("sky_scan") or {}).get("pass") if plan.get("sky_scan") else "n/a (no sky classes)",
        run_checks=plan.get("run_checks"),
        run_checks_short=dict((k, (plan.get("run_checks") or {}).get(k)) for k in
                              ("layout", "n_frames_with_pose", "n_views", "coasted_pct", "path_length_m", "time_monotonic")),
        recipe={k: v for k, v in rec.items() if k not in CFG.OPERATIONAL_KEYS}, config_hash=plan.get("config_hash"),
        frame=("map frame = the run's world frame w (T_world_cam0_rectified, OpenCV camera axes; metres)"),
        licence_note=("the map is derived from your sequence: its licence follows your data's licence; xclean-maps itself "
                      "is Apache-2.0"),
        privacy=CFG.PRIVACY_NOTE)
    U.atomic_json(md + "/report.json", rep)
    EXR.report_md(md + "/report.md", rep)
    if a.web:
        web_export(md, a.splat_transform, log)
    if not rescore["pass_"]:
        log("export: the re-score from splats.ply differs from the trained model by %.4f dB (> 0.02 dB)" % d)
        return False
    log("export done: %d splats (%d invisible removed), %.1f MB; re-score diff %.4f dB; held-out PSNR %.2f dB (splats on "
        "black); files in %s" % (sinfo["n"], cinfo["n_invisible_removed"], sinfo["bytes"] / 1e6, d,
                                 al.get("psnr_raw_nosky") or float("nan"), md))
    return True


def web_export(md, tool, log):
    """OPTIONAL (off by default; untested in xclean-maps): splats.ply -> map.sog with PlayCanvas splat-transform
    (`splat-transform` on PATH, or --splat-transform "node path/to/cli.mjs")."""
    import shlex
    import subprocess
    cmd = shlex.split(tool or "splat-transform", posix=(os.name != "nt")) + ["-w", "-q", md + "/splats.ply", md + "/map.sog"]
    log("web export: %s" % " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log("web export FAILED (rc %d): %s" % (p.returncode, (p.stderr or p.stdout)[-1500:]))
        return False
    return True


RUNNERS = dict(depth=stage_depth, prep=stage_prep, train=stage_train, eval=stage_eval, export=stage_export)


def build(a):
    """The `xclean-maps build` command."""
    dirs = make_dirs(a.out)
    U.set_write_root(dirs["out"])
    U.ensure_dir(dirs["work"])
    U.ensure_dir(dirs["hot"])
    log = U.Log(os.path.join(dirs["work"], "build.log"), prefix="[xclean-maps] ")
    spec = resolve_spec(a.run_dir, a.sequence, a.out, a.family, a.preset, a.limit, a.steps, a.ext, a.name,
                        gate=not a.no_gate, depth_evidence=a.depth_evidence)
    old = U.read_json(dirs["work"] + "/build_spec.json")
    U.atomic_json(dirs["work"] + "/build_spec.json", spec)
    if old and old != spec:
        log("build spec changed since the last build in this folder: stages whose inputs changed are re-run")
    rec = CFG.recipe(spec["preset"])
    if a.max_chunk_s:
        rec["max_chunk_s"] = float(a.max_chunk_s)
    log("map %s: run %s, sequence %s, family %s, preset %s" % (spec["name"], spec["run_dir"], spec["sequence"],
                                                                spec["family"], spec["preset"]))
    want = [s for s in STAGES if s in (a.stages.split(",") if a.stages else STAGES)]
    if a.device == "cuda" and any(s in want for s in ("train", "eval", "export")):
        setup_device("cuda", rec, log)
    elif a.device == "cpu":
        setup_device("cpu", rec, log)
    t0 = time.time()
    for st in want:
        h = stage_hash(spec, dirs, st, rec)
        if not a.force and marker_ok(dirs, st, h):
            log("stage %s: up to date, skipped" % st)
            continue
        clear_marker(dirs, st)
        ts = time.time()
        log("stage %s: start" % st)
        try:
            ok = RUNNERS[st](spec, dirs, rec, a, log)
        except Exception as e:
            if type(e).__name__ == "KillRule":
                log("training stopped by kill rule %s (checkpoint kept). Review work/train.log, then re-run with "
                    "--ack-kill %s to continue, or with --fresh to start over" % (e.rule, e.rule))
                return 3
            raise
        if st == "train":
            h = stage_hash(spec, dirs, st, rec)
        mark(dirs, st, h, ok, dict(seconds=round(time.time() - ts, 1)))
        log("stage %s: %s in %.0f s" % (st, "done" if ok else "NOT finished", time.time() - ts))
        if not ok:
            if st == "train":
                log("training stopped before the end (STOP file, chunk limit or --stop-after); run the same command "
                    "again to resume")
            return 1
    log("build finished in %.0f s: %s" % (time.time() - t0, dirs["out"]))
    if a.clean_cache and os.path.isdir(dirs["hot"]):
        U.remove_if_exists(dirs["hot"])
        for st in ("depth", "prep"):
            clear_marker(dirs, st)
        log("cache removed (--clean-cache): a later re-run starts again from the depth stage")
    return 0
