# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""CPU tests of xclean-maps (no GPU, no gsplat CUDA kernel is run).

Run:  CUDA_VISIBLE_DEVICES=-1 python -I -B -m unittest discover -s tests -v

Tests that need real data read it READ-ONLY from the paths in the environment variables below and SKIP when they are
not set (the defaults are the maintainers' machine; they are not part of the package):
  XCLEAN_MAPS_TEST_RUN       an xclean replay/live run folder         (default: the OAK 2026-03-27 16:40:02 board run)
  XCLEAN_MAPS_TEST_SEQ       the sequence folder it ran on
  XCLEAN_MAPS_TEST_REF_HOT   the internal builder's prepared arrays of the same run (identity test of PREP)
  XCLEAN_MAPS_TEST_REF_WORK  the internal builder's work dir of the same run (its plan.json)
  XCLEAN_MAPS_TEST_KIT_RUN   an `xclean run`-style output folder with kit/trajectory_kf.csv
Outputs go to XCLEAN_MAPS_TEST_OUT (default <repo>/test_out).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src", "xclean_maps")
OUT = os.path.abspath(os.environ.get("XCLEAN_MAPS_TEST_OUT") or os.path.join(REPO, "test_out"))
RUN = os.environ.get("XCLEAN_MAPS_TEST_RUN", "D:/xoneslam_runs/xclean/board_oak_20260327_164002")
SEQ = os.environ.get("XCLEAN_MAPS_TEST_SEQ", "E:/wm_oak/oak_20260327_164002")
REF_HOT = os.environ.get("XCLEAN_MAPS_TEST_REF_HOT", "D:/xclean_maps_v1_hot/oak_20260327_164002__board")
REF_WORK = os.environ.get("XCLEAN_MAPS_TEST_REF_WORK", "E:/xclean_maps_v1/work/oak_20260327_164002__board")
KIT_RUN = os.environ.get("XCLEAN_MAPS_TEST_KIT_RUN", "D:/xoneslam_runs/xclean/c1_ver2_mh_04_difficult")

from xclean_maps import util as U  # noqa: E402

U.add_write_root(OUT)
os.makedirs(OUT, exist_ok=True)


def _have(*paths):
    return all(p and os.path.exists(p) for p in paths)


class TestImportAndCli(unittest.TestCase):
    def test_imports(self):
        import importlib
        for m in ("cli", "config", "util", "io_lib", "runs", "sources", "masks", "frame_quality", "scale", "losses",
                  "exposure", "sky", "render", "mcmc_wrap", "visible_adam", "sampler", "pose_refine", "live", "prep",
                  "train", "evaluate", "export", "pipeline", "viewer"):
            importlib.import_module("xclean_maps." + m)

    def _cli(self, *args):
        exe = shutil.which("xclean-maps", path=os.path.dirname(sys.executable))
        cmd = [exe] if exe else [sys.executable, "-I", "-m", "xclean_maps"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        return subprocess.run(cmd + list(args), capture_output=True, text=True, env=env, timeout=300)

    def test_help(self):
        for args in (("--help",), ("build", "--help"), ("view", "--help"), ("info", "--help")):
            p = self._cli(*args)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("usage", p.stdout)
        p = self._cli("build", "--help")
        for opt in ("--sequence", "--out", "--preset", "--device"):
            self.assertIn(opt, p.stdout)


class TestNoHardCodedPaths(unittest.TestCase):
    """No drive path, user folder or internal tool path in the package source (only CLI options / --out)."""

    def test_grep(self):
        bad = re.compile(r"(\b[A-Za-z]:[/\\])|claudius|xclean_maps_v1_hot|xclean_web_assets|third_party_repro|node\.exe|"
                         r"GPU_JOB\.lock|rows_xclean\.json|compactslam|dreamsim|xsesto\.tools")
        hits = []
        for fn in sorted(os.listdir(SRC)):
            if not fn.endswith(".py"):
                continue
            for i, ln in enumerate(open(os.path.join(SRC, fn), encoding="utf-8"), 1):
                if bad.search(ln):
                    hits.append("%s:%d: %s" % (fn, i, ln.strip()[:120]))
        self.assertEqual(hits, [], "\n".join(hits))

    def test_config_has_no_paths(self):
        from xclean_maps import config as CFG
        txt = json.dumps(dict(r=CFG.RECIPE, f=CFG.FAMILY, p=CFG.PRESETS), default=str)
        self.assertIsNone(re.search(r"[A-Za-z]:[/\\]", txt))

    def test_recipe_unchanged_from_internal(self):
        """RECIPE equals the internal website-map recipe except max_chunk_s (operational)."""
        ref = os.environ.get("XCLEAN_MAPS_TEST_INTERNAL_CONFIG", "E:/xclean_maps_v1/builder/config.py")
        if not os.path.isfile(ref):
            self.skipTest("internal config not available")
        import ast
        tree = ast.parse(open(ref, encoding="utf-8").read())
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "RECIPE")
        src = ast.get_source_segment(open(ref, encoding="utf-8").read(), node.value)
        internal = eval(src, {"dict": dict})
        from xclean_maps import config as CFG
        diff = {k for k in set(internal) | set(CFG.RECIPE) if internal.get(k) != CFG.RECIPE.get(k)}
        self.assertEqual(diff, {"max_chunk_s"})
        self.assertEqual(CFG.recipe("default"), CFG.RECIPE)


@unittest.skipUnless(_have(RUN), "real replay run folder not available")
class TestRunReader(unittest.TestCase):
    def test_replay_run(self):
        from xclean_maps import runs as RUNS
        r = RUNS.open_run(RUN)
        self.assertEqual(r.layout, "replay")
        c = r.no_gt_checks()
        self.assertGreater(c["n_frames_with_pose"], 0)
        self.assertGreater(c["n_views"], 0)
        self.assertTrue(c["time_monotonic"])
        self.assertEqual(c["n_bad_rotation"], 0)
        for f in r.kf_frames[:50]:
            R = r.T[f][:3, :3]
            self.assertLess(np.abs(R.T @ R - np.eye(3)).max(), 1e-4)
        ref = os.path.join(REF_HOT, "depth", "depth_index.json")
        if os.path.isfile(ref):                       # identity with the internal builder's view list
            self.assertEqual(r.kf_frames, json.load(open(ref))["frames"])
        rp = os.path.join(REF_WORK, "plan.json")
        if os.path.isfile(rp):
            self.assertEqual(r.stream, json.load(open(rp))["pose_stream"])

    def test_info_cli(self):
        exe = shutil.which("xclean-maps", path=os.path.dirname(sys.executable))
        args = ["info", RUN] + (["--sequence", SEQ] if _have(SEQ) else [])
        p = subprocess.run(([exe] if exe else [sys.executable, "-I", "-m", "xclean_maps"]) + args, capture_output=True,
                           text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr)
        d = json.loads(p.stdout)
        self.assertEqual(d["layout"], "replay")


@unittest.skipUnless(_have(os.path.join(KIT_RUN, "kit", "trajectory_kf.csv")), "no xclean-run kit folder available")
class TestKitReader(unittest.TestCase):
    def test_kit_run(self):
        from xclean_maps import runs as RUNS
        r = RUNS.open_run(KIT_RUN)
        self.assertEqual(r.layout, "run")
        self.assertGreater(len(r.kf_frames), 10)
        c = r.no_gt_checks()
        self.assertTrue(c["time_monotonic"])
        for f in r.kf_frames[:20]:
            self.assertLess(np.abs(r.T[f][:3, :3].T @ r.T[f][:3, :3] - np.eye(3)).max(), 1e-4)


@unittest.skipUnless(_have(RUN, SEQ, os.path.join(REF_HOT, "depth", "depth_full.npy")),
                     "real run + sequence + internal SGM32 depth not available")
class TestPrepFewFrames(unittest.TestCase):
    """PREP on 8 keyframes of a real OAK run. The depth stage (SGM32) needs a GPU, so its output is taken from the
    internal builder's depth array of the same run (read-only), rows picked for the same 8 frames. Then the per-frame
    images and mask bits must equal the internal builder's prepared arrays (identity of the maths), apart from the bits
    that depend on the neighbouring views (moving / static-world), which differ by construction with 8 views."""

    N = 8

    @classmethod
    def setUpClass(cls):
        from xclean_maps import pipeline as PL
        from xclean_maps import prep as PX
        from xclean_maps import runs as RUNS
        from xclean_maps import config as CFG
        cls.out = os.path.join(OUT, "prep8")
        if os.path.isdir(cls.out):
            shutil.rmtree(cls.out)
        cls.dirs = PL.make_dirs(cls.out)
        U.add_write_root(cls.out)
        cls.spec = PL.resolve_spec(RUN, SEQ, cls.out, preset="small", limit=cls.N)
        run = RUNS.open_run(RUN)
        frames = PX.select_views(run, cls.N)
        ref_idx = json.load(open(os.path.join(REF_HOT, "depth", "depth_index.json")))
        pos = [ref_idx["frames"].index(f) for f in frames]
        ref = np.load(os.path.join(REF_HOT, "depth", "depth_full.npy"), mmap_mode="r")
        P = PX.depth_paths(cls.dirs)
        U.ensure_dir(P["dir"])
        arr = np.ascontiguousarray(ref[pos])
        np.save(P["arr"], arr)
        has = [bool((a > 0).any()) for a in arr]
        U.atomic_json(P["index"], dict(row="prep8", version=PX.DEPTH_VERSION, depth_source="sgm32", licence_clean=True,
                                       contract=ref_idx.get("contract"), frames=[int(f) for f in frames], has=has,
                                       sha1=[U.sha1_bytes(np.ascontiguousarray(a).tobytes()) for a in arr],
                                       valid_share=[float((a > 0).mean()) for a in arr],
                                       valid_share_stats=U.stats([float((a > 0).mean()) for a in arr]), stats=None,
                                       image_size=[arr.shape[2], arr.shape[1]], fB=ref_idx.get("fB"), run_dir=RUN,
                                       xclean_modules=None, array_sha1=U.sha1_file(P["arr"]),
                                       test_note="rows of the internal builder's SGM32 array (GPU stage not run on CPU)"))
        cls.ref_pos = pos
        cls.plan = PX.prepare(cls.spec, cls.dirs, CFG.recipe("small"), workers=4, limit=cls.N)
        cls.prep = PX.Prepared(cls.dirs, "prep8")

    def test_plan(self):
        p = self.plan
        self.assertEqual(p["n_views"], self.N)
        self.assertEqual(p["image_size"], [640, 400])
        self.assertEqual(p["channels"], 1)
        self.assertEqual(p["pose_stream"], "LIVE-MAP")
        self.assertGreater(p["seeds"]["n_total"], 1000)
        self.assertEqual(p["family"], "oak")
        for k in ("img", "disp", "mask"):
            self.assertTrue(os.path.isfile(os.path.join(self.dirs["hot"], k + ".npy")))

    def test_identity_with_internal_prep(self):
        if not os.path.isfile(os.path.join(REF_HOT, "img.npy")):
            self.skipTest("internal prepared arrays not available")
        from xclean_maps.losses import BIT_MOVING
        ri = np.load(os.path.join(REF_HOT, "img.npy"), mmap_mode="r")
        rm = np.load(os.path.join(REF_HOT, "mask.npy"), mmap_mode="r")
        rd = np.load(os.path.join(REF_HOT, "disp.npy"), mmap_mode="r")
        for j, q in enumerate(self.ref_pos):
            self.assertTrue(np.array_equal(self.prep.img[j], ri[q]), "image of view %d differs" % j)
            self.assertTrue(np.array_equal(self.prep.disp[j], rd[q]), "disparity of view %d differs" % j)
            a = np.asarray(self.prep.mask[j]) & ~np.uint8(BIT_MOVING)
            b = np.asarray(rm[q]) & ~np.uint8(BIT_MOVING)
            self.assertTrue(np.array_equal(a, b), "mask bits of view %d differ: %d px" % (j, int((a != b).sum())))
        rp = os.path.join(REF_WORK, "plan.json")
        if os.path.isfile(rp):
            ref_plan = json.load(open(rp))
            np.testing.assert_allclose(self.plan["up"], ref_plan["up"], atol=1e-9)
            np.testing.assert_allclose(self.plan["K_rect"], ref_plan["K_rect"], atol=1e-9)
            self.assertAlmostEqual(self.plan["fB"], ref_plan["fB"], places=9)

    def test_cameras_json(self):
        from xclean_maps import export as EXR
        p = os.path.join(self.out, "cameras.json")
        EXR.cameras_json(p, self.prep)
        cams = json.load(open(p))
        self.assertEqual(len(cams), self.N)
        T = np.array(cams[0]["T_world_cam"])
        np.testing.assert_allclose(np.array(cams[0]["rotation"]), T[:3, :3], atol=1e-8)
        np.testing.assert_allclose(np.array(cams[0]["position"]), T[:3, 3], atol=1e-6)

    def test_keyframes_and_pointcloud(self):
        from xclean_maps import export as EXR
        k = EXR.keyframes(os.path.join(self.out, "keyframes"), self.prep, n=4)
        self.assertEqual(k["n"], 4)
        pc = EXR.pointcloud(os.path.join(self.out, "pointcloud.ply"), self.prep, self.plan["near_clip_m"],
                            self.plan["max_depth_m"])
        self.assertGreater(pc["points"], 100)


class TestPly(unittest.TestCase):
    def test_round_trip(self):
        from xclean_maps import export as EXR
        from xclean_maps import io_lib as IO
        rng = np.random.RandomState(0)
        n = 5000
        params = dict(means=rng.randn(n, 3).astype(np.float32), scales=rng.randn(n, 3).astype(np.float32) - 3,
                      quats=rng.randn(n, 4).astype(np.float32), opacities=rng.randn(n).astype(np.float32),
                      sh0=rng.randn(n, 1, 3).astype(np.float32), shN=(0.1 * rng.randn(n, 3, 3)).astype(np.float32))
        params["opacities"][:100] = -20.0                     # invisible: removed by clean_params
        clean, info = EXR.clean_params(params, channels=3)
        self.assertEqual(info["n_invisible_removed"], 100)
        d = os.path.join(OUT, "ply")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "splats.ply")
        s = EXR.write_splats(path, clean)
        self.assertTrue(s["schema_ok"] and s["own_reader_exact"] and s["plyfile_check"] in (True, None))
        back = IO.params_from_splat_ply(path)
        self.assertTrue(np.array_equal(back["means"], clean["means"]))
        self.assertTrue(np.array_equal(back["scales"], clean["scales"]))
        self.assertTrue(np.array_equal(back["opacities"], clean["opacities"]))
        self.assertTrue(np.allclose(back["shN"], clean["shN"]))
        q = clean["quats"] / np.linalg.norm(clean["quats"], axis=1, keepdims=True)
        self.assertTrue(np.allclose(back["quats"], q))
        try:
            from plyfile import PlyData                       # optional independent parser (GPL-3.0, not a dependency)
            v = PlyData.read(path)["vertex"]
            self.assertEqual([p.name for p in v.properties], IO.SPLAT_PLY_FIELDS)
            self.assertTrue(s["plyfile_check"])
        except ImportError:
            pass
        # grey maps: channels averaged
        g, gi = EXR.clean_params(params, channels=1)
        self.assertTrue(gi["grey_equalised"])
        self.assertTrue(np.allclose(g["sh0"][..., 0], g["sh0"][..., 2]))

    def test_viewer_snapshot_points(self):
        """The viewer's CPU fallback renders a map folder to a PNG (no window)."""
        from xclean_maps import export as EXR
        from xclean_maps import viewer as V
        rng = np.random.RandomState(1)
        n = 20000
        pts = np.c_[rng.uniform(-5, 5, n), rng.uniform(-1, 1, n), rng.uniform(4, 20, n)].astype(np.float32)
        params = dict(means=pts, scales=np.full((n, 3), -3, np.float32), quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
                      opacities=np.full(n, 3.0, np.float32), sh0=rng.randn(n, 1, 3).astype(np.float32),
                      shN=np.zeros((n, 3, 3), np.float32))
        md = os.path.join(OUT, "viewmap")
        os.makedirs(md, exist_ok=True)
        EXR.write_splats(os.path.join(md, "splats.ply"), params)
        K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]])
        views = [dict(j=0, kf_id=0, frame_index=0, t=0.0, role="train", T_wc=np.eye(4).tolist())]
        prep = SimpleNamespace(K=K, W=320, H=240, views=views, T_wc=lambda j: np.eye(4))
        EXR.cameras_json(os.path.join(md, "cameras.json"), prep)
        png = os.path.join(md, "snap.png")
        r = V.view(md, mode="points", snapshot=png)
        self.assertTrue(os.path.isfile(png))
        import cv2
        im = cv2.imread(png)
        self.assertEqual(im.shape, (240, 320, 3))
        self.assertGreater(int((im.max(2) > 12).sum()), 1000)
        self.assertIn("points", r["renderer"])


class TestWriteFence(unittest.TestCase):
    def test_refuses_outside_root(self):
        with self.assertRaises(PermissionError):
            U.assert_writable(os.path.join(os.path.dirname(OUT), "outside_" + str(os.getpid()), "x.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
