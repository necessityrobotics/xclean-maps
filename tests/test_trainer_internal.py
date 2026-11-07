"""(Ported from the internal builder's tests/test_trainer.py; imports and the scratch dir changed only.)
The trainer end to end on CPU (torch reference rasteriser, gsplat ops with CPU reference kernels):
0b(6) exposure identity no-op, 0b(7) sky alpha = 1 no-op, 0b(8) disparity loss 0 on an exact map,
0b(9) checkpoint -> resume byte for byte (parameters, VisibleAdam + row-step state, MCMC state, exposure, sky, sampler,
RNGs) through relocations and re-seeding, the loss going down, and the live preview writer."""
from __future__ import annotations

import json
import os
import shutil
import unittest

import numpy as np
import torch

from xclean_maps import config as CFG
from xclean_maps import exposure as EX
from xclean_maps import io_lib as IO
from xclean_maps import losses as LS
from xclean_maps import mcmc_wrap as MW
from xclean_maps import sky as SKY
from xclean_maps.train import PARAM_NAMES, DataSource, Trainer
from xclean_maps.render import reference_renderer
from synthetic import SyntheticPrepared

torch.set_num_threads(1)
from xclean_maps import util as _U

SCR = os.path.join(os.environ.get("XCLEAN_MAPS_TEST_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_out"), "trainer")
os.makedirs(SCR, exist_ok=True)
_U.add_write_root(SCR)


def _recipe(S):
    r = CFG.recipe()
    r.update(coverage_every_frac=0.25, refine_every=20, live_every_init=40, live_every_min=10, live_width=320,
             sky_w=64, sky_h=32, ckpt_every_s=1e9, sh_ramp_steps=60,
             k1_enabled=False)          # toy scene: K1 is exercised by its own test below, not by the mechanics tests
    return r


def _dirs(name):
    d = os.path.join(SCR, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    return os.path.join(d, "work"), os.path.join(d, "hot")


def _trainer(prep, name, S, live=False):
    work, hot = _dirs(name)
    return Trainer(prep, _recipe(S), "cpu", reference_renderer(antialiased=True), work, hot, steps=S, live=live,
                   ckpt_every_s=1e9, milestones=(0.5, 1.0))


class TestNoOps(unittest.TestCase):
    def test_exposure_identity_is_noop(self):
        e = EX.Exposure(5, 3)
        img = torch.rand(6, 7, 3)
        self.assertTrue(torch.equal(e.apply(img, 2), img.clamp(0, 1)))
        g = EX.Exposure(5, 1)
        self.assertTrue(torch.equal(g.apply(img, 4), img.clamp(0, 1)))
        self.assertEqual(float(e.prior(1)), 0.0)

    def test_sky_alpha_one_is_noop(self):
        s = SKY.SkyPanorama([0, -1.0, 0], 3, 64, 32, init=0.7)
        rgb = torch.rand(6, 7, 3)
        al = torch.ones(6, 7, 1)
        d = SKY.ray_dirs_world(np.eye(4), np.array([[5.0, 0, 3.5], [0, 5.0, 3.0], [0, 0, 1]]), 7, 6)
        self.assertTrue(torch.equal(SKY.composite(rgb, al, s.sample(d)), rgb))

    def test_disparity_loss_zero_on_exact_map(self):
        fB = 41.68
        z = torch.rand(10, 12) * 20 + 0.5
        disp = fB / z
        mask = torch.ones(10, 12, dtype=torch.bool)
        l, n = LS.disparity_loss(z, torch.ones(10, 12), disp, mask, fB)
        self.assertEqual(n, 120)
        self.assertLess(float(l), 1e-10)


class TestTrainer(unittest.TestCase):
    def test_loss_decreases_and_live_preview(self):
        """Mechanics (loss down, relocations, live preview, milestones) with the exposure fit OFF: on this 32x24 toy the
        render stays near 15 dB, so any per-photo exposure fit mostly absorbs render error (phase-2a A/B,
        _scratch/toy_ab2.py: exposure off 16.8 dB; robust closed form from 0.3 S 15.3 dB; from 0.1 S unmasked 14.6 dB).
        test_exposure_does_not_break_training covers the default exposure path."""
        prep = SyntheticPrepared(steps=240)
        S = 240
        tr = _trainer(prep, "loss_live", S, live=True)
        tr.rec["exposure_start_frac"] = 2.0
        data = DataSource(prep, "cpu", resident=False)

        def train_psnr():
            ps = []
            for j in prep.train_ids:
                rgb, _, _ = tr.render_view(j, data)
                ps.append(float(-10 * torch.log10(((rgb - data.get(j)[0]) ** 2).mean())))
            return float(np.mean(ps))
        tr.setup_fresh()
        p0 = train_psnr()
        with MW.cpu_reference_kernels():
            info = tr.run(data, resume=False)
        p1 = train_psnr()
        probes = [c["probe"] for c in tr.curve if c.get("probe") is not None]
        print("  synthetic: training-view PSNR %.2f -> %.2f dB, probe %s, final N %d, relocations %d, reseeds %s, "
              "live previews %d (every %s, overhead %s)"
              % (p0, p1, np.round(probes, 2), info["n_gaussians"], info["relocations"], [r["moved"] for r in info["reseeds"]],
                 info["live_previews"], info["live_every"], info["live_overhead_frac"]))
        self.assertGreater(p1, p0 + 3.0)
        self.assertGreater(info["relocations"], 0)
        self.assertGreater(info["live_previews"], 0)
        st = json.load(open(os.path.join(tr.work, "live", "stats.json")))
        for k in ("step", "total_steps", "n_gaussians", "loss", "heldout_probe_psnr", "reserved_gib", "allocated_gib",
                  "it_per_s", "eta_s", "row", "stage"):
            self.assertIn(k, st)
        import cv2
        im = cv2.imread(os.path.join(tr.work, "live", "preview.jpg"))
        self.assertIsNotNone(im)
        self.assertTrue(os.path.isfile(os.path.join(tr.work, "ckpt", "milestone_050.pt")))
        self.assertTrue(os.path.isfile(os.path.join(tr.work, "ckpt", "milestone_100.pt")))

    def test_exposure_does_not_break_training(self):
        """The default (robust, gauge-normalised, from 0.3 S) exposure path on the toy: training still improves by
        >= 1.5 dB at identity exposure, and the exposure common to all photos stays the identity (gauge)."""
        prep = SyntheticPrepared(steps=240)
        S = 240
        tr = _trainer(prep, "expo_on", S)
        data = DataSource(prep, "cpu", resident=False)

        def train_psnr():
            return float(np.mean([float(-10 * torch.log10(((tr.render_view(j, data)[0] - data.get(j)[0]) ** 2).mean()))
                                  for j in prep.train_ids]))
        tr.setup_fresh()
        p0 = train_psnr()
        with MW.cpu_reference_kernels():
            tr.run(data, resume=False)
        p1 = train_psnr()
        n = [tr.expo.normalised(i)[0] for i in range(len(prep.train_ids)) if not tr.expo.is_identity(i)]   # fitted photos
        self.assertGreater(len(n), 0)
        gm = float(torch.exp(torch.log(torch.stack(n)).mean()))
        print("  exposure on: training-view PSNR %.2f -> %.2f dB; normalised gains %.3f-%.3f, geometric mean %.6f"
              % (p0, p1, float(torch.stack(n).min()), float(torch.stack(n).max()), gm))
        self.assertGreater(p1, p0 + 1.5)
        self.assertAlmostEqual(gm, 1.0, places=5)

    def test_resume_is_byte_exact(self):
        """0b(9): run A uninterrupted; run B stops after 130 steps (rolling checkpoint) and a NEW trainer resumes it."""
        S = 220
        prep = SyntheticPrepared(steps=S, seed=3)
        with MW.cpu_reference_kernels():
            ta = _trainer(prep, "resume_a", S)
            ta.run(DataSource(prep, "cpu", False), resume=False)
            tb = _trainer(prep, "resume_b", S)
            out = tb.run(DataSource(prep, "cpu", False), resume=False, stop_after=130)
            self.assertTrue(out["stopped"])
            ck = tb.rolling_checkpoints()
            self.assertEqual(len(ck), 1)
            # a brand-new trainer object (as after a reboot) on the same dirs
            tc = Trainer(prep, _recipe(S), "cpu", reference_renderer(antialiased=True), tb.work, tb.hot, steps=S, live=False,
                         ckpt_every_s=1e9, milestones=(0.5, 1.0))
            tc.run(DataSource(prep, "cpu", False), resume=True)
        self.assertEqual(ta.step, tc.step)
        self.assertGreater(len(ta.strategy.history), 0)
        self.assertGreater(sum(r["moved"] for r in ta.reseed_log), 0)
        for n in PARAM_NAMES:
            self.assertTrue(torch.equal(ta.params[n].detach(), tc.params[n].detach()), n)
            sa, sc = ta.opts[n].state[ta.params[n]], tc.opts[n].state[tc.params[n]]
            for k in ("exp_avg", "exp_avg_sq", "row_step"):
                self.assertTrue(torch.equal(sa[k], sc[k]), "%s %s" % (n, k))
        self.assertTrue(torch.equal(ta.expo.param.detach(), tc.expo.param.detach()))
        self.assertTrue(torch.equal(ta.sky.param.detach(), tc.sky.param.detach()))
        self.assertEqual(ta.sampler.state_dict(), tc.sampler.state_dict())
        ra, rc = ta.checkpoint_blob()["rng"], tc.checkpoint_blob()["rng"]       # review N1: the two runs' states
        self.assertTrue(torch.equal(ra["torch"], rc["torch"]))
        self.assertEqual(ra["py"], rc["py"])
        self.assertTrue(np.array_equal(ra["np"][1], rc["np"][1]))
        la = [c["loss"] for c in ta.curve]
        lc = [c["loss"] for c in tc.curve]
        self.assertEqual(la, lc)                                    # a continuous loss trace through the planned kill
        print("  resume: %d steps, %d gaussians, %d relocations, %d re-seed passes: byte-identical"
              % (ta.step, len(ta.params["means"]), len(ta.strategy.history), len(ta.reseed_log)))

    def test_k1_fires(self):
        S = 40
        prep = SyntheticPrepared(steps=S, seed=5)
        tr = _trainer(prep, "k1", S)
        tr.rec["k1_enabled"] = True
        tr.setup_fresh()
        from xclean_maps.train import KillRule
        tr.k1_check(rec_probe=20.0)
        with self.assertRaises(KillRule):
            tr.k1_check(rec_probe=18.9)                              # probe down >= 1 dB
        tr.setup_fresh()
        tr.strategy.history = [dict(step=s, n_before=1000, relocated=60, added=0, n_after=1000) for s in (100, 200, 300)]
        with self.assertRaises(KillRule):
            tr.k1_check()                                            # dead share > 5 % for 3 refines
        tr.setup_fresh()
        tr.k1_check(cov_full=[dict(j=1, kf_id=101, coverage=0.97)])
        with self.assertRaises(KillRule):
            tr.k1_check(cov_full=[dict(j=1, kf_id=101, coverage=0.85)])   # coverage down >= 0.10

    def test_k1_enabled_resume_is_byte_exact(self):
        """review N1 / S4: with K1 ON, a resumed run equals the uninterrupted one (the K1 loss window is checkpointed)."""
        S = 160
        prep = SyntheticPrepared(steps=S, seed=9)
        r = _recipe(S)
        r["k1_enabled"] = True
        from xclean_maps.train import KillRule

        def go(t, **kw):
            try:
                t.run(DataSource(prep, "cpu", False), **kw)
                return None
            except KillRule as e:                                  # a kill must happen identically in both runs
                return (e.rule, t.step)
        with MW.cpu_reference_kernels():
            wa, ha = _dirs("k1res_a")
            ta = Trainer(prep, r, "cpu", reference_renderer(antialiased=True), wa, ha, steps=S, live=False, ckpt_every_s=1e9,
                         milestones=(1.0,))
            ka = go(ta, resume=False)
            wb, hb = _dirs("k1res_b")
            tb = Trainer(prep, r, "cpu", reference_renderer(antialiased=True), wb, hb, steps=S, live=False, ckpt_every_s=1e9,
                         milestones=(1.0,))
            go(tb, resume=False, stop_after=90)
            tc = Trainer(prep, r, "cpu", reference_renderer(antialiased=True), wb, hb, steps=S, live=False, ckpt_every_s=1e9,
                         milestones=(1.0,))
            kc = go(tc, resume=True)
        print("  K1-enabled resume: uninterrupted %s, resumed %s, step %d / %d, loss window %d entries"
              % (ka or "no kill", kc or "no kill", ta.step, tc.step, len(ta.loss_win)))
        self.assertEqual(ka, kc)
        self.assertEqual(ta.step, tc.step)
        self.assertEqual(list(ta.loss_win), list(tc.loss_win))
        for n in PARAM_NAMES:
            self.assertTrue(torch.equal(ta.params[n].detach(), tc.params[n].detach()), n)

    def test_stop_file_stops_cleanly(self):
        S = 400
        prep = SyntheticPrepared(steps=S, seed=6)
        tr = _trainer(prep, "stopfile", S)
        os.makedirs(tr.work, exist_ok=True)
        open(os.path.join(tr.work, "STOP"), "w").close()
        with MW.cpu_reference_kernels():
            out = tr.run(DataSource(prep, "cpu", False), resume=False)
        self.assertTrue(out["stopped"])
        self.assertEqual(out["step"], 100)
        self.assertEqual(len(tr.rolling_checkpoints()), 1)
        os.remove(os.path.join(tr.work, "STOP"))

    def test_stale_checkpoint_refused(self):
        S = 60
        prep = SyntheticPrepared(steps=S, seed=4)
        with MW.cpu_reference_kernels():
            tb = _trainer(prep, "stale", S)
            tb.run(DataSource(prep, "cpu", False), resume=False, stop_after=30)
            r2 = _recipe(S)
            r2["w_bce"] = 0.07                     # a different config must never silently resume an old checkpoint
            tc = Trainer(prep, r2, "cpu", reference_renderer(), tb.work, tb.hot, steps=S, live=False, ckpt_every_s=1e9)
            with self.assertRaises(SystemExit):
                tc.run(DataSource(prep, "cpu", False), resume=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
