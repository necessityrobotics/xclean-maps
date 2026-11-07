# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""xclean-maps command line.

  xclean-maps build <xclean output folder> [--sequence <sequence folder>] --out <map folder> [--preset small|default|full]
  xclean-maps view  <map folder> [--renderer auto|gsplat|points] [--snapshot file.png]
  xclean-maps info  <xclean output folder> [--sequence <sequence folder>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _build_parser():
    ap = argparse.ArgumentParser(prog="xclean-maps", description="Gaussian-splat maps from xclean runs (version %s)"
                                 % __version__)
    ap.add_argument("--version", action="version", version="xclean-maps %s" % __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a map: depth -> prep -> train -> eval -> export",
                       description="Build a 3D Gaussian-splat map from an xclean output folder (`xclean replay` / "
                                   "`xclean live` run folder, or `xclean run` output folder) and the sequence folder it "
                                   "ran on. Run the same command again to continue after a stop.")
    b.add_argument("run_dir", help="the xclean output folder (recorder.jsonl + p1_frames.jsonl, or kit/trajectory_kf.csv)")
    b.add_argument("--sequence", default=None, help="the sequence folder the run was made from (default: the folder "
                                                    "the run recorded, when it recorded one)")
    b.add_argument("--out", required=True, help="the map folder to write (everything goes in here)")
    b.add_argument("--preset", default="default", choices=("small", "default", "full"),
                   help="small = quick test budget; default = the recipe of the xclean website maps; full = default "
                        "+ the diagnostic eval passes (default: %(default)s)")
    b.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="cuda (needed for a real map) or cpu "
                                                                              "(tiny tests only)")
    b.add_argument("--family", default="auto", help="camera family (auto = from calib.json 'source'): oak, euroc, kitti, "
                                                    "4seasons, madmax, rosario, rosariov2, tartanair")
    b.add_argument("--name", default=None, help="map name in the reports (default: the --out folder name)")
    b.add_argument("--stages", default=None, help="comma list of stages to run (default: all: depth,prep,train,eval,export)")
    b.add_argument("--limit", type=int, default=None, help="use only this many keyframes, evenly spaced (tests)")
    b.add_argument("--steps", type=int, default=None, help="override the number of training steps (tests)")
    b.add_argument("--ext", default=None, help="image extension of the sequence (default: detected, png or jpg)")
    b.add_argument("--workers", type=int, default=4, help="CPU threads for prep (default %(default)s)")
    b.add_argument("--fresh", dest="resume", action="store_false", help="start training from scratch (default: resume "
                                                                        "from the newest checkpoint of this build)")
    b.add_argument("--ack-kill", default=None, help="continue a training that a kill rule stopped (e.g. K1), after review")
    b.add_argument("--stop-after", type=int, default=None, help="stop training after N steps of this session (tests)")
    b.add_argument("--max-chunk-s", type=float, default=None, help="stop training after this many seconds (checkpointed; "
                                                                   "run again to continue)")
    b.add_argument("--force", action="store_true", help="re-run stages even when they are up to date")
    b.add_argument("--no-live", action="store_true", help="no live preview images during training")
    b.add_argument("--no-gate", action="store_true", help="skip xclean's dataset integrity gate when opening the sequence")
    b.add_argument("--depth-evidence", action="store_true", help="also store SGM32's cost-volume evidence arrays "
                                                                 "(only the non-default sky rules read them)")
    b.add_argument("--web", action="store_true", help="OPTIONAL: also write map.sog with PlayCanvas splat-transform")
    b.add_argument("--splat-transform", default=None, help='the splat-transform command (default "splat-transform")')
    b.add_argument("--clean-cache", action="store_true", help="delete <out>/cache after a finished build")

    v = sub.add_parser("view", help="open a fly-through window on a built map")
    v.add_argument("map_dir")
    v.add_argument("--renderer", default="auto", choices=("auto", "gsplat", "points"))
    v.add_argument("--snapshot", default=None, help="render the first camera to this PNG and exit (no window)")
    v.add_argument("--browser", action="store_true",
                   help="open the map in your web browser (three.js + Spark, served from this PC only; Ctrl+C stops it)")
    v.add_argument("--port", type=int, default=0, help="--browser: port on 127.0.0.1 (default: a free port)")
    v.add_argument("--no-open", action="store_true", help="--browser: print the address, do not open a browser")

    i = sub.add_parser("info", help="read an xclean output folder and print what a build would use (no writes)")
    i.add_argument("run_dir")
    i.add_argument("--sequence", default=None)
    i.add_argument("--family", default="auto")
    return ap


def cmd_info(a):
    from . import pipeline as PL
    from . import runs as RUNS
    run = RUNS.open_run(a.run_dir)
    seq = a.sequence or run.sequence_folder()
    fam = a.family if a.family != "auto" else (PL.family_from_calib(seq) if seq else None)
    out = dict(run_dir=run.dir, layout=run.layout, pose_stream=run.stream, sequence=seq, family=fam,
               checks=run.no_gt_checks(), gravity_rule=run.gravity()[1])
    print(json.dumps(out, indent=1, default=str))
    return 0


def main(argv=None):
    a = _build_parser().parse_args(argv)
    if a.cmd == "build":
        if a.device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        from . import pipeline
        return pipeline.build(a)
    if a.cmd == "view":
        if a.browser:
            from . import webview
            return webview.serve(a.map_dir, port=a.port, open_browser=not a.no_open)
        from . import viewer
        r = viewer.view(a.map_dir, a.renderer, a.snapshot)
        if a.snapshot:
            print(json.dumps(r))
        return 0
    if a.cmd == "info":
        return cmd_info(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
