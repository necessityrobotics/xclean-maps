# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""train -> eval -> export -> view of the pipeline on CPU, on a tiny synthetic scene (torch reference rasteriser,
gsplat MCMC ops with the CPU reference kernels). Checks the stage wiring and every output file; it says nothing about
map quality (that needs the GPU test on real data)."""
from __future__ import annotations

import json
import os
import shutil
import unittest
from types import SimpleNamespace

import numpy as np

from xclean_maps import util as U

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.environ.get("XCLEAN_MAPS_TEST_OUT") or os.path.join(os.path.dirname(HERE), "test_out"))


class TestPipelineSynthetic(unittest.TestCase):
    def test_train_eval_export_view(self):
        import torch
        from synthetic import SyntheticPrepared
        from xclean_maps import config as CFG
        from xclean_maps import mcmc_wrap as MW
        from xclean_maps import pipeline as PL
        torch.set_num_threads(2)
        out = os.path.join(OUT, "pipeline_synth")
        if os.path.isdir(out):
            shutil.rmtree(out)
        dirs = PL.make_dirs(out)
        U.set_write_root(OUT)
        U.ensure_dir(dirs["work"])
        U.ensure_dir(dirs["hot"])
        sp = SyntheticPrepared(n_views=16, steps=120, cap=600)
        for k in ("img", "disp", "mask"):
            np.save(os.path.join(dirs["hot"], k + ".npy"), getattr(sp, k))
        np.savez(os.path.join(dirs["hot"], "seeds.npz"), **sp.seeds)
        U.atomic_json(os.path.join(dirs["hot"], "views.json"), dict(row="synth", views=sp.views))
        plan = dict(sp.plan, K_rect=sp.K.tolist(), image_size=[sp.W, sp.H], channels=sp.channels, fB=sp.fB,
                    up=sp.up.tolist(), max_depth_m=50.0, inputs_gb=0.0, inputs={}, n_views=len(sp.views),
                    n_train=len(sp.train_ids), n_hold=len(sp.hold_ids), pose_stream="synthetic",
                    run_checks=dict(layout="synthetic"), sky_scan=None)
        U.atomic_json(os.path.join(dirs["work"], "plan.json"), plan)
        rec = CFG.recipe("default")
        rec.update(coverage_every_frac=0.25, refine_every=20, sky_w=64, sky_h=32, ckpt_every_s=1e9, sh_ramp_steps=60,
                   live_width=320, k1_enabled=False)
        spec = dict(name="synth", preset="default", run_dir="(synthetic)", sequence="(synthetic)", family="tartanair")
        a = SimpleNamespace(device="cpu", workers=1, resume=True, stop_after=None, no_live=False, ack_kill=None, web=False,
                            splat_transform=None, oblique_size=(64, 36))
        log = U.Log(os.path.join(dirs["work"], "build.log"))
        with MW.cpu_reference_kernels():
            self.assertTrue(PL.stage_train(spec, dirs, rec, a, log))
            self.assertTrue(PL.stage_eval(spec, dirs, rec, a, log))
            self.assertTrue(PL.stage_export(spec, dirs, rec, a, log))
        for fn in ("splats.ply", "cameras.json", "pointcloud.ply", "preview.png", "report.json", "report.md"):
            self.assertTrue(os.path.isfile(os.path.join(out, fn)), fn)
        self.assertTrue(os.listdir(os.path.join(out, "previews")))
        self.assertTrue(os.path.isfile(os.path.join(out, "keyframes", "keyframes.json")))
        rep = json.load(open(os.path.join(out, "report.json")))
        self.assertTrue(rep["rescore"]["pass_"])
        self.assertEqual(rep["heldout"]["n"], len(sp.hold_ids))
        self.assertGreater(rep["heldout"]["psnr_splats_on_black_db"], 5.0)
        self.assertEqual(len(json.load(open(os.path.join(out, "cameras.json")))), len(sp.views))
        self.assertTrue(os.path.isfile(os.path.join(dirs["work"], "live", "preview.jpg")))
        from xclean_maps import viewer as V
        r = V.view(out, mode="points", snapshot=os.path.join(out, "snap.png"))
        self.assertTrue(os.path.isfile(os.path.join(out, "snap.png")))
        self.assertGreater(r["n_splats"], 10)
        log.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
