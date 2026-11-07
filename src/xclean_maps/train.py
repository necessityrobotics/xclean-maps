# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""TRAIN (BUILDER 5): ONE gsplat model per whole route, trained jointly from scratch. No chunks, no merge (A6).

MCMC with a fixed budget (noise_lr 5e5 / L^2, destination-zeroing relocate), VisibleAdam on rows with a nonzero
data-term gradient, regularisers on visible rows only, epoch-shuffled sampler, disparity Huber depth loss (+ a bounded
one-sided near-clip penalty), clip-aware photometric loss, closed-form per-image exposure, stereo sky / sky-maybe masks +
panorama, coverage re-seeding (round-robin over the whole route), hard 10 m size clamp, antialiased rasterisation.
Atomic checkpoints (rolling every 30 min in <out>/cache/ckpt tagged with a RUN ID, milestones at 25/50/75/100 % in
<out>/work/ckpt, fsynced and sha1-verified), byte-exact resume (parameters, optimiser + row-step state, MCMC state, exposure, sky, sampler, all RNGs,
the K1 loss window), status.json heartbeat, live preview (A5), kill rules K1/K2/K5/K6, STOP file.

Phase 2a review fixes: M1 (coverage targets disjoint), M2 (run id; a fresh run clears the stale run's checkpoints and
logs; resume only the current run; drop newer / invalid checkpoints), M3 (the config hash includes a digest of the PREP
inputs, verified against the files at start), S1 (status / curve writes never kill the run), S2 (closed-form exposure),
S3 (near-clip penalty), S4 (K1 window checkpointed), S5 (K2), S7 (re-seeding round-robin), N2 (K5 pause, K6 estimate from
the run config), N3 (fsync before replace), N5 (fy; circular sky; milestone before the K1 coverage check; operational
knobs outside the hash).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from . import pose_refine as PR

from . import config as CFG
from . import exposure as EX
from . import io_lib as IO
from . import losses as LS
from . import mcmc_wrap as MW
from . import sky as SKY
from . import util as U
from .live import LivePreview, overview_camera
from .sampler import EpochSampler
from .visible_adam import VisibleAdam

PARAM_NAMES = ("means", "scales", "quats", "opacities", "sh0", "shN")
TRAINER_VERSION = "onego-v4.3"
# knobs that change how the run is operated, not what it computes: never part of the checkpoint hash (review N5d)
OPERATIONAL_KEYS = CFG.OPERATIONAL_KEYS
# K1 low-alpha corroboration (lead amendment 2026-10-19): constants in code, not recipe keys (the checkpoint hash stays)
K1_LOW_ALPHA_RISE, K1_LOW_ALPHA_PROBE_DB, K1_LOW_ALPHA_COV = 0.10, 0.5, 0.05
# K1 500-step-loss sub-rule, amended like the low-alpha one (lead 2026-10-22, THRESHOLDS 10.9): a >= 30 % rise counts as
# collapse only when CORROBORATED by the probe >= 0.5 dB below its best or the coverage median >= 0.05 below its best
K1_LOSS500_RISE = 1.30
# depth trust (THRESHOLDS 10.9, oak_20260329_161834 kf 5769-5784): when the static-world test rejects more than half of a
# view, its RAFT disparity is inconsistent with its neighbours and none of it is used (no depth loss, no re-seeding)
MAX_MOVING_SHARE_FOR_DEPTH = 0.5


def reseed_scale(z, fx, stride_px, z_near_m, far_px):
    """Initial isotropic sigma (m) of a coverage re-seed at RAFT depth z. Near field (z <= z_near_m = fB / seed_min_disp_px,
    the depth limit of the initial seeding): half the re-seed stride footprint, z * stride / fx * 0.5 (unchanged).
    Far field (phase 2c arm, eval 2026-10-19: oversized far re-seeds bled into the sky): far_px pixels' footprint at the
    true depth, z * far_px / fx, clamped to at most the near-field size at the boundary, z_near_m * stride / fx * 0.5."""
    near = z * stride_px / fx * 0.5
    far = torch.minimum(z * far_px / fx, torch.full_like(z, z_near_m * stride_px / fx * 0.5))
    return torch.where(z > z_near_m, far, near).clamp_min(1e-3)


def delta_inv_detached(d):
    with torch.no_grad():
        return PR.delta_inv(d)


def sky_photo_stopgrad(C, rgb, alphas, sky_rgb, sky_px):
    """Phase 2c (2026-10-22, work/_arms): on CONFIDENT-sky pixels the photometric loss reaches only the panorama, never the
    splats. The image is unchanged (same forward value); splats there receive only the sky BCE (alpha -> 0). Evidence:
    sky-coloured splats near the horizon fit the sky better than the direction-only panorama (view-dependent SH colour),
    so the photometric reward kept them against the BCE on 137 of 161 training views (SKY-6 p95 > 0.10)."""
    Cs = rgb.detach() + (1.0 - alphas.detach()) * sky_rgb
    return torch.where(sky_px[..., None], Cs, C)


class KillRule(Exception):
    def __init__(self, rule, detail):
        super().__init__("%s: %s" % (rule, detail))
        self.rule, self.detail = rule, detail


# --------------------------------------------------------------------------------------------- input digest (M3)
def _sha1_arr(a):
    h = hashlib.sha1()
    a = np.ascontiguousarray(a)
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def prep_input_digest(prep, verify_files=True):
    """Digest of everything the model is trained from: the view list (poses + roles, from memory), the seeds, the PREP
    input hashes (kit poses sha1, image / cache digests, depth key, calib) and the img / disp / mask arrays. For a
    prepared row the array digests are the sha1s PREP recorded, re-checked against the files (a stale or edited PREP
    refuses to train); for in-memory test data they are computed from the arrays."""
    views = [(int(v["j"]), int(v["kf_id"]), [float(x) for r in v["T_wc"] for x in r], v["role"], bool(v.get("probe")))
             for v in prep.views]
    d = dict(views=U.sha1_json(views), seeds={k: _sha1_arr(v) for k, v in sorted(prep.seeds.items())},
             plan_inputs=(prep.plan.get("inputs") or {}), K=np.asarray(prep.K, float).round(12).tolist(), fB=float(prep.fB),
             size=[int(prep.W), int(prep.H)])
    files = getattr(prep, "array_files", None)
    if files:
        inp = prep.plan.get("inputs") or {}
        for key, path in files.items():
            rec = inp.get(key + "_npy_sha1")
            if verify_files:
                cur = U.sha1_file(path)
                if rec and cur != rec:
                    raise SystemExit("PREP input %s changed since PREP (sha1 %s != plan %s): re-run PREP" % (path, cur, rec))
                d[key] = cur
            else:
                d[key] = rec
    else:
        d.update(img=_sha1_arr(prep.img), disp=_sha1_arr(prep.disp), mask=_sha1_arr(prep.mask))
    return U.sha1_json(d), d


# --------------------------------------------------------------------------------------------- data on the device
class DataSource:
    """Serves one view's tensors. GPU-resident when the inputs fit (BUILDER 6: 1.4 x model + inputs <= 18 GiB), else
    pinned host RAM copied per step (non_blocking). On CPU it is plain memory."""

    def __init__(self, prep, device, resident):
        self.device = torch.device(device)
        self.prep = prep
        ids = sorted(set(prep.train_ids) | set(prep.hold_ids))
        self.pos = {j: i for i, j in enumerate(ids)}
        sel = np.array(ids, int)
        img = torch.from_numpy(np.ascontiguousarray(prep.img[sel]))
        disp = torch.from_numpy(np.ascontiguousarray(prep.disp[sel]))
        mask = torch.from_numpy(np.ascontiguousarray(prep.mask[sel]))
        self.resident = bool(resident) and self.device.type == "cuda"
        if self.resident:
            img, disp, mask = img.to(self.device), disp.to(self.device), mask.to(self.device)
        elif self.device.type == "cuda":
            img, disp, mask = img.pin_memory(), disp.pin_memory(), mask.pin_memory()
        self.img, self.disp, self.mask = img, disp, mask
        self.T_wc = {j: prep.T_wc(j) for j in ids}
        self.T_cw = {j: torch.tensor(IO.inv_T(self.T_wc[j]), dtype=torch.float32, device=self.device) for j in ids}
        # review MF2: no per-view direction fields (8-12 GiB on the big rows); ONE camera-frame pixel-ray grid per camera
        # model + each view's 3x3 rotation on the device; dirs = grid @ R^T per step (bit-identical to the old fields)
        self.R_wc = {j: torch.tensor(self.T_wc[j][:3, :3], dtype=torch.float32, device=self.device) for j in ids}
        self._grid = {}
        self.channels = prep.channels

    def get(self, j):
        i = self.pos[j]
        nb = self.device.type == "cuda"
        img = self.img[i].to(self.device, non_blocking=nb).float() / 255.0
        if self.channels == 1:
            img = img.expand(-1, -1, 3)
        disp = self.disp[i].to(self.device, non_blocking=nb).float()
        mask = self.mask[i].to(self.device, non_blocking=nb)
        return img, disp, mask, self.T_cw[j]

    def dirs(self, j, K, W, H):
        """World ray directions of view j: the cached camera-frame grid rotated by the view's R_wc (no host copy)."""
        key = (int(W), int(H), tuple(np.round(np.asarray(K, np.float64).ravel(), 9).tolist()))
        g = self._grid.get(key)
        if g is None:
            g = SKY.ray_dirs_world(np.eye(4), K, W, H, self.device)
            self._grid[key] = g
        return g @ self.R_wc[j].T

    def maybe_proxy_sky(self, j):
        """2c (2026-10-22): the maybe-sky pixels the SKY-6 gate counts as sky (BIT_SKY_MAYBE AND the scan's colour / grey
        sky proxy, eval_row.eval_heldout's rule, computed from the photo and the SLAM pose exactly as there). Computed once
        per view on first use and cached on the device as bool."""
        c = getattr(self, "_mps", None)
        if c is None:
            c = self._mps = {}
        t = c.get(j)
        if t is None:
            from .masks import proxy_sky, ray_elevation
            u8 = np.asarray(self.prep.img[j])
            u8 = u8[..., 0] if (u8.ndim == 3 and u8.shape[2] == 1) else u8
            b = np.asarray(self.prep.mask[j])
            el = ray_elevation(self.T_wc[j], self.prep.K, self.prep.W, self.prep.H, self.prep.up)
            m = ((b & LS.BIT_SKY_MAYBE) != 0) & proxy_sky(u8, el > -np.sin(np.radians(3.0)))
            t = c[j] = torch.from_numpy(np.ascontiguousarray(m)).to(self.device)
        return t

    def maybe_proxy_sky_interior(self, j, edge_px=7):
        """2c (2026-10-22, factory row): the maybe-proxy-sky pixels farther than edge_px from any non-SKY-6 pixel - the
        part of the maybe set that SKY-6 decides on (eval_row.sky6_split). Cached on the device as bool."""
        c = getattr(self, "_mpsi", None)
        if c is None:
            c = self._mpsi = {}
        t = c.get(j)
        if t is None:
            import cv2
            m = self.maybe_proxy_sky(j).cpu().numpy()
            s6 = m | ((np.asarray(self.prep.mask[j]) & LS.BIT_SKY) != 0)
            dist = cv2.distanceTransform(s6.astype(np.uint8), cv2.DIST_L2, 3) if s6.any() else np.zeros(s6.shape, np.float32)
            t = c[j] = torch.from_numpy(np.ascontiguousarray(m & (dist > edge_px))).to(self.device)
        return t

    def cache_bytes(self):
        return int(sum(g.numel() * g.element_size() for g in self._grid.values()) + 36 * len(self.R_wc)
                   + sum(t.numel() for t in (getattr(self, "_mps", None) or {}).values())
                   + sum(t.numel() for t in (getattr(self, "_mpsi", None) or {}).values()))

    def img_numpy(self, j):
        a = self.prep.img[j].astype(np.float32) / 255.0
        return np.repeat(a, 3, axis=2) if a.shape[2] == 1 else a


def init_params(seeds, device, sh_degree=1, init_opacity=0.5):
    X = torch.tensor(seeds["xyz"], dtype=torch.float32, device=device)
    N = len(X)
    scales = torch.tensor(seeds["log_scale"], dtype=torch.float32, device=device)[:, None].repeat(1, 3)
    quats = torch.zeros((N, 4), device=device)
    quats[:, 0] = 1.0
    opac = torch.logit(torch.full((N,), float(init_opacity), device=device))
    sh0 = ((torch.tensor(seeds["rgb"], dtype=torch.float32, device=device) - 0.5) / IO.SH_C0)[:, None, :]
    shN = torch.zeros((N, (sh_degree + 1) ** 2 - 1, 3), device=device)
    return torch.nn.ParameterDict({"means": torch.nn.Parameter(X), "scales": torch.nn.Parameter(scales),
                                   "quats": torch.nn.Parameter(quats), "opacities": torch.nn.Parameter(opac),
                                   "sh0": torch.nn.Parameter(sh0), "shN": torch.nn.Parameter(shN)})


def _fsync_replace(tmp, dst):
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    U._replace_retry(tmp, dst)


# --------------------------------------------------------------------------------------------- the trainer
class Trainer:
    def __init__(self, prep, recipe, device, renderer, out_work, out_hot, log=None, steps=None, estimate_s=None,
                 live=True, ckpt_every_s=None, milestones=None, stage_name="train", probe_n=None, coverage=True,
                 run_config=None, verify_inputs=True):
        self.prep = prep
        self.rec = recipe
        self.device = torch.device(device)
        self.render = renderer
        self.work = U.ensure_dir(out_work)
        self.hot = U.ensure_dir(out_hot)
        self.ckdir = U.ensure_dir(os.path.join(out_hot, "ckpt"))
        self.msdir = U.ensure_dir(os.path.join(out_work, "ckpt"))
        self.log = log or U.Log(os.path.join(out_work, "train.log"), prefix="[train %s] " % prep.row)
        plan = prep.plan
        self.L = float(plan["L"])
        self.cap = int(plan["cap_max"])
        self.S = int(steps or plan["steps"])
        self.n_train = len(prep.train_ids)
        self.run_config = dict(run_config or {})
        est_h = self.run_config.get("estimate_h")
        self.estimate_s = estimate_s if estimate_s is not None else (float(est_h) * 3600 if est_h else None)
        self.probe_step_s = (float(self.run_config["probe_step_ms"]) / 1e3) if self.run_config.get("probe_step_ms") else None
        self.k2_db = self.run_config.get("k2_db")
        self.ckpt_every_s = float(ckpt_every_s if ckpt_every_s is not None else recipe["ckpt_every_s"])
        self.milestones = tuple(milestones if milestones is not None else recipe["milestones"])
        self.K = torch.tensor(prep.K, dtype=torch.float32, device=self.device)
        self.Knp = np.asarray(prep.K, float)
        self.W, self.H = int(prep.W), int(prep.H)
        self.fB = prep.fB
        self.near = float(plan.get("near_clip_m") or 0.05)
        self.win = IO.gauss_win(device=self.device)
        self.stage_name = stage_name
        self.want_live = live
        self.coverage_on = coverage
        self.probe_ids = prep.probe_ids[: (probe_n or recipe["probe_views"])]
        self.input_digest, self.input_parts = prep_input_digest(prep, verify_files=verify_inputs)
        math_recipe = {k: v for k, v in recipe.items() if k not in OPERATIONAL_KEYS}
        self.config_hash = CFG.config_hash(dict(plan_hash=plan.get("config_hash"), recipe=math_recipe, S=self.S,
                                                trainer=TRAINER_VERSION, renderer=getattr(renderer, "kind", "?"),
                                                inputs=self.input_digest))
        self.t_train = np.array([prep.views[j]["t"] for j in prep.train_ids], float)
        self._ti = {j: i for i, j in enumerate(prep.train_ids)}
        self.poser = None
        if recipe.get("pose_opt", False):
            self.poser = PR.PoseRefiner(len(prep.train_ids), device, int(recipe["pose_opt_start_frac"] * self.S), self.S,
                                        lr_rot_deg=recipe["pose_lr_rot_deg"], lr_trans_m=recipe["pose_lr_trans_m"],
                                        w_prior=recipe["pose_w_prior"], s_rot_deg=recipe["pose_s_rot_deg"],
                                        s_trans_m=recipe["pose_s_trans_m"], cap_rot_deg=recipe["pose_cap_rot_deg"],
                                        cap_trans_m=recipe["pose_cap_trans_m"])
        self.run_id = None
        self.events = []                  # (kind, step) of probe / reseed / coverage, in loop order (tests)

    # ---------------------------------------------------------------- run identity (M2)
    @property
    def run_file(self):
        return os.path.join(self.hot, "run.json")

    def start_fresh_run(self):
        """Stale-artefact rule for a fresh run: the previous run's rolling checkpoints, kill record, logs and outputs are
        removed BEFORE setup; its milestones are moved aside (kept for post-mortem, never resumed)."""
        old = U.read_json(self.run_file) or {}
        for p in glob.glob(os.path.join(self.ckdir, "*.pt")) + glob.glob(os.path.join(self.ckdir, "*.tmp*")):
            U.remove_if_exists(p)
        for fn in ("kill.json", "curve.csv", "coverage.csv", "status.json", "train_summary.json", "final_params.pt"):
            U.remove_if_exists(os.path.join(self.work, fn))
        U.remove_if_exists(os.path.join(self.work, "live"))
        ms = [p for p in glob.glob(os.path.join(self.msdir, "milestone_*")) if os.path.isfile(p)]
        if ms:
            dst = U.ensure_dir(os.path.join(self.msdir, "superseded_%s" % (old.get("run_id") or "unknown")))
            for p in ms:
                os.replace(p, os.path.join(dst, os.path.basename(p)))
        self.run_id = time.strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]
        U.atomic_json(self.run_file, dict(run_id=self.run_id, config_hash=self.config_hash, input_digest=self.input_digest,
                                          started=time.strftime("%Y-%m-%dT%H:%M:%S"), superseded=old.get("run_id")))

    # ---------------------------------------------------------------- setup
    def _lrs(self):
        r = self.rec
        return dict(means=r["lr_means_per_L"] * self.L, scales=r["lr_scales"], quats=r["lr_quats"],
                    opacities=r["lr_opacities"], sh0=r["lr_sh0"], shN=r["lr_shN"])

    def _make_optimizers(self):
        r = self.rec
        lrs = self._lrs()
        self.lr0 = lrs
        self.opts = {n: VisibleAdam([{"params": [self.params[n]], "lr": lrs[n], "name": n}], lr=lrs[n],
                                    betas=r["adam_betas"], eps=r["adam_eps"]) for n in PARAM_NAMES}
        self.opt_sky = VisibleAdam([{"params": [self.sky.param], "lr": r["lr_sky"], "name": "sky"}], lr=r["lr_sky"],
                                   betas=r["adam_betas"], eps=1e-15)

    def _make_strategy(self):
        r = self.rec
        self.strategy = MW.MCMCFixedBudget(cap_max=self.cap, noise_lr=MW.noise_lr_metric(self.L, r["noise_lr_normalised"]),
                                           refine_start_iter=self.n_train, refine_stop_iter=int(r["refine_stop_frac"] * self.S),
                                           refine_every=r["refine_every"], min_opacity=r["min_opacity"], verbose=False)
        self.sstate = self.strategy.initialize_state()

    def setup_fresh(self):
        r = self.rec
        seed = int(r["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.params = init_params(self.prep.seeds, self.device, r["sh_degree"], r["init_opacity"])
        self.expo = EX.Exposure(self.n_train, 1 if self.prep.channels == 1 else 3, device=self.device)
        self.sky = SKY.SkyPanorama(self.prep.up, 1 if self.prep.channels == 1 else 3, r["sky_w"], r["sky_h"], device=self.device)
        self.sky.min_lat_deg = self.rec.get("sky_pano_min_lat_deg")
        self._make_optimizers()
        self._make_strategy()
        self.sampler = EpochSampler(self.n_train, seed=seed)
        self.step = 0
        self.curve, self.coverage, self.telemetry, self.k1 = [], [], [], dict(best_probe=None)
        self.loss_win = []
        self.wall_prev = 0.0
        self.reseed_log = []
        self.events = []
        self.mem_frozen_at = None

    # ---------------------------------------------------------------- checkpoint
    def _opt_state(self, opt):
        out = []
        for g in opt.param_groups:
            for p in g["params"]:
                st = opt.state.get(p, {})
                out.append({k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in st.items()})
        return dict(lr=[g["lr"] for g in opt.param_groups], state=out)

    def _load_opt_state(self, opt, blob):
        for g, lr in zip(opt.param_groups, blob["lr"]):
            g["lr"] = lr
        ps = [p for g in opt.param_groups for p in g["params"]]
        for p, st in zip(ps, blob["state"]):
            opt.state[p] = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in st.items()}

    def checkpoint_blob(self):
        rng = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(self.device)
        return dict(
            version=TRAINER_VERSION, config_hash=self.config_hash, input_digest=self.input_digest, run_id=self.run_id,
            step=int(self.step), S=self.S, row=self.prep.row,
            params={n: self.params[n].detach().cpu().clone() for n in PARAM_NAMES},
            opts={n: self._opt_state(o) for n, o in self.opts.items()},
            expo=self.expo.param.detach().cpu().clone(),
            sky=self.sky.param.detach().cpu().clone(), opt_sky=self._opt_state(self.opt_sky),
            strategy=dict(binoms=self.sstate["binoms"].detach().cpu().clone(), cap_max=self.strategy.cap_max,
                          refine_stop_iter=self.strategy.refine_stop_iter, history=list(self.strategy.history),
                          last=dict(self.strategy.last)),
            sampler=self.sampler.state_dict(), rng=rng, loss_win=list(self.loss_win),
            curve=list(self.curve), coverage=list(self.coverage), telemetry=list(self.telemetry), k1=json.loads(json.dumps(self.k1)),
            reseed_log=list(self.reseed_log), mem_frozen_at=self.mem_frozen_at,
            poser=(self.poser.state_dict() if self.poser is not None else None),
            wall_s=self.wall_prev + (time.time() - self.t_session0 if hasattr(self, "t_session0") else 0.0))

    def save_checkpoint(self, path):
        blob = self.checkpoint_blob()
        U.assert_writable(path)
        tmp = path + ".tmp%d" % os.getpid()
        torch.save(blob, tmp)
        _fsync_replace(tmp, path)
        return path

    def restore(self, blob):
        if blob.get("config_hash") != self.config_hash:
            raise SystemExit("checkpoint config hash %s != current %s (stale checkpoint: other config or other PREP inputs; "
                             "start fresh without --resume)" % (blob.get("config_hash"), self.config_hash))
        self.params = torch.nn.ParameterDict({n: torch.nn.Parameter(blob["params"][n].to(self.device)) for n in PARAM_NAMES})
        ch = 1 if self.prep.channels == 1 else 3
        self.expo = EX.Exposure(self.n_train, ch, device=self.device)
        self.expo.param = torch.nn.Parameter(blob["expo"].to(self.device), requires_grad=False)
        self.sky = SKY.SkyPanorama(self.prep.up, ch, self.rec["sky_w"], self.rec["sky_h"], device=self.device)
        self.sky.param = torch.nn.Parameter(blob["sky"].to(self.device))
        self.sky.min_lat_deg = self.rec.get("sky_pano_min_lat_deg")
        self._make_optimizers()
        for n, o in self.opts.items():
            self._load_opt_state(o, blob["opts"][n])
        self._load_opt_state(self.opt_sky, blob["opt_sky"])
        self._make_strategy()
        st = blob["strategy"]
        self.sstate["binoms"] = st["binoms"].to(self.device)
        self.strategy.cap_max = st["cap_max"]
        self.strategy.refine_stop_iter = st["refine_stop_iter"]
        self.strategy.history = list(st["history"])
        self.strategy.last = dict(st["last"])
        self.sampler = EpochSampler.from_state(blob["sampler"])
        r = blob["rng"]
        random.setstate(r["py"])
        np.random.set_state(r["np"])
        torch.set_rng_state(r["torch"])
        if self.device.type == "cuda" and "cuda" in r:
            torch.cuda.set_rng_state(r["cuda"], self.device)
        self.step = int(blob["step"])
        self.curve, self.coverage, self.telemetry = list(blob["curve"]), list(blob["coverage"]), list(blob["telemetry"])
        self.k1 = blob["k1"]
        self.loss_win = list(blob.get("loss_win", []))
        self.reseed_log = list(blob.get("reseed_log", []))
        self.mem_frozen_at = blob.get("mem_frozen_at")
        if self.poser is not None:
            if blob.get("poser") is None:
                raise SystemExit("checkpoint has no pose-refinement state but the recipe has pose_opt on")
            self.poser.load_state_dict(blob["poser"])
        self.wall_prev = float(blob.get("wall_s", 0.0))
        self.run_id = blob.get("run_id")

    @staticmethod
    def valid_blob(blob):
        try:
            for n in PARAM_NAMES:
                t = blob["params"][n]
                if not torch.isfinite(t).all():
                    return False
            n0 = blob["params"]["means"].shape[0]
            return all(blob["params"][n].shape[0] == n0 for n in PARAM_NAMES)
        except Exception:
            return False

    @staticmethod
    def _ck_step(p):
        return int(os.path.basename(p).rsplit("_", 1)[1][:-3])

    def rolling_checkpoints(self, run_id=None):
        rid = run_id if run_id is not None else self.run_id
        if rid is None:
            return []
        fs = glob.glob(os.path.join(self.ckdir, "ckpt_%s_*.pt" % rid))
        return sorted(fs, key=self._ck_step)

    def _kill_gate(self, rid):
        """Review SF1: a kill is sticky. kill.json of the current run refuses the resume until the lead acknowledges that
        exact rule (xclean-maps build --resume --ack-kill <rule>); the acknowledged record is kept (renamed), never deleted."""
        kp = os.path.join(self.work, "kill.json")
        k = U.read_json(kp)
        if not k or k.get("run_id") != rid:
            return
        ovp = self.run_config.get("k1_override")
        if ovp and k.get("rule") == "K1" and os.path.isfile(ovp):
            ov = U.read_json(ovp) or {}
            if ov.get("run_id") == rid and int(ov.get("kill_step", -1)) == int(k.get("step", -2)) and ov.get("rule") == "K1" \
                    and str(ov.get("countersigned_by") or "").strip():
                ov["_path"] = ovp.replace("\\", "/")
                self._k1_override_pending = (k, ov)
                self.log("K1 kill at step %s: lead override %s found; the amended K1 is re-evaluated after the restore" % (k.get("step"), ovp))
                return
            raise SystemExit("K1 override %s does not match this kill (run %s step %s) or is not countersigned" % (ovp, rid, k.get("step")))
        ack = self.run_config.get("ack_kill")
        if ack and str(ack) == str(k.get("rule")):
            dst = os.path.join(self.work, "kill_acked_%s_%s.json" % (k.get("rule"), time.strftime("%Y%m%d%H%M%S")))
            U._replace_retry(kp, dst)
            self.log("kill %s at step %s ACKNOWLEDGED (--ack-kill %s): resuming; record kept as %s" % (k.get("rule"), k.get("step"), ack, dst))
            return
        raise SystemExit("run %s was stopped by kill rule %s at step %s (%s): review it, then resume with --ack-kill %s, "
                         "or start fresh without --resume" % (rid, k.get("rule"), k.get("step"), k.get("detail"), k.get("rule")))

    @staticmethod
    def _corrupt(e):
        import pickle
        import zipfile
        if isinstance(e, (EOFError, pickle.UnpicklingError, zipfile.BadZipFile)):
            return True
        m = str(e)
        return isinstance(e, RuntimeError) and any(x in m for x in ("PytorchStreamReader", "zip archive", "central directory",
                                                                    "invalid header", "unexpected EOF"))

    def try_resume(self):
        """Resume only the CURRENT run (hot/run.json): its newest valid rolling checkpoint (key / shape / finite / run id
        checks), else the previous one. Checkpoints of OTHER runs are removed (M2). Review SF2: a load error is retried
        once; only a verified-corrupt file (unpickling / EOF / zip errors, or failing the checks) is SET ASIDE as *.bad,
        never deleted; any other error (memory, permission, I/O) stops with SystemExit and touches nothing; if the run had
        checkpoints and none loads, SystemExit (never a silent fresh start). A kill of this run is sticky (SF1)."""
        info = U.read_json(self.run_file)
        if not info or not info.get("run_id"):
            return None
        rid = info["run_id"]
        self._kill_gate(rid)
        for p in glob.glob(os.path.join(self.ckdir, "ckpt_*.pt")):
            if not os.path.basename(p).startswith("ckpt_%s_" % rid):
                self.log("dropping a checkpoint of another run: %s" % p)
                U.remove_if_exists(p)
        cands = self.rolling_checkpoints(rid)
        loaded = None
        for p in reversed(cands):
            blob, err = None, None
            for attempt in (1, 2):
                try:
                    blob = torch.load(p, map_location="cpu", weights_only=False)
                    err = None
                    break
                except Exception as e:      # noqa: BLE001
                    err = e
                    self.log("checkpoint %s: load attempt %d failed (%r)" % (p, attempt, e))
                    time.sleep(2.0)
            if err is not None:
                if self._corrupt(err):
                    U._replace_retry(p, p + ".bad")
                    self.log("checkpoint %s is corrupt: set aside as %s.bad (not deleted)" % (p, p))
                    continue
                raise SystemExit("checkpoint %s could not be read (%r); nothing was changed: fix the cause (memory, a file "
                                 "lock) and resume again" % (p, err))
            if not self.valid_blob(blob) or blob.get("run_id") != rid:
                U._replace_retry(p, p + ".bad")
                self.log("checkpoint %s failed the key / shape / finite / run-id checks: set aside as .bad (not deleted)" % p)
                continue
            self.restore(blob)
            loaded = p
            pend = getattr(self, "_k1_override_pending", None)
            if pend:
                self.k1_reevaluate_override(*pend)
                self._k1_override_pending = None
            break
        if loaded is None:
            if cands:
                raise SystemExit("no loadable checkpoint of run %s (%d set aside as *.bad in %s): inspect them, or start "
                                 "fresh without --resume" % (rid, len(cands), self.ckdir))
            return None
        for p in self.rolling_checkpoints(rid):
            if self._ck_step(p) > self._ck_step(loaded):
                U._replace_retry(p, p + ".newer_than_loaded")
        self.log("RESUMED run %s from %s at step %d (%d gaussians)" % (rid, loaded, self.step, len(self.params["means"])))
        return loaded

    def _rolling(self):
        p = os.path.join(self.ckdir, "ckpt_%s_%09d.pt" % (self.run_id, self.step))
        self.save_checkpoint(p)
        for old in self.rolling_checkpoints()[:-int(self.rec["ckpt_keep"])]:
            U.remove_if_exists(old)
        return p

    def _milestone(self, frac):
        name = "milestone_%03d.pt" % int(round(frac * 100))
        hot = os.path.join(self.ckdir, name)
        self.save_checkpoint(hot)
        dst = os.path.join(self.msdir, name)
        sha = U.sha1_file(hot)
        tmp = dst + ".tmp"
        U.assert_writable(tmp)
        shutil.copyfile(hot, tmp)
        _fsync_replace(tmp, dst)                 # review N3: fsync on E: before the replace
        if U.sha1_file(dst) != sha:
            raise RuntimeError("milestone copy sha1 mismatch: %s" % dst)
        U.remove_if_exists(hot)
        U.atomic_json(dst + ".json", dict(step=self.step, sha1=sha, bytes=os.path.getsize(dst), frac=frac, run_id=self.run_id,
                                          n_gaussians=len(self.params["means"])))
        self.log("milestone %d %%: %s (sha1 %s verified)" % (int(round(frac * 100)), dst, sha[:12]))

    # ---------------------------------------------------------------- schedule helpers
    def lr_means(self, step):
        return self.lr0["means"] * self.rec["lr_means_final_frac"] ** min(step / max(self.S, 1), 1.0)

    def _every(self, frac):
        return max(1, int(round(frac * self.S)))

    def vram(self):
        if self.device.type != "cuda":
            return dict(reserved_gib=0.0, allocated_gib=0.0, max_reserved_gib=0.0, max_allocated_gib=0.0)
        g = 2 ** 30
        return dict(reserved_gib=torch.cuda.memory_reserved(self.device) / g, allocated_gib=torch.cuda.memory_allocated(self.device) / g,
                    max_reserved_gib=torch.cuda.max_memory_reserved(self.device) / g,
                    max_allocated_gib=torch.cuda.max_memory_allocated(self.device) / g)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ---------------------------------------------------------------- one training step
    def train_step(self, data, timing=False):
        r = self.rec
        step = self.step
        tt = {}
        t = time.perf_counter()
        ti = self.sampler.next()
        j = self.prep.train_ids[ti]
        img, disp, mask, Tcw = data.get(j)
        if timing:
            self.sync(); tt["loader"] = time.perf_counter() - t; t = time.perf_counter()
        sh_deg = min(step // r["sh_ramp_steps"], r["sh_degree"])
        pose_d = None
        if self.poser is not None and self.poser.active(step):
            Tcw, pose_d = self.poser.apply(ti, Tcw)
        colors, alphas, info = self.render(self.params, Tcw, self.K, self.W, self.H, sh_deg, depth=True)
        rgb, ed, al = colors[..., :3], colors[..., 3], alphas[..., 0]
        dirs = data.dirs(j, self.Knp, self.W, self.H)
        if pose_d is not None:
            with torch.no_grad():
                Rw = data.R_wc[j]
                dirs = dirs @ (Rw @ self.poser.R_cam(ti) @ Rw.T).T      # the panorama looks along the refined camera
        sky_rgb = self.sky.sample(dirs)
        C = SKY.composite(rgb, alphas, sky_rgb)
        b = mask
        sky_px = (b & LS.BIT_SKY) != 0
        if r.get("sky6_maybe_as_sky", False):
            sky_px = sky_px | data.maybe_proxy_sky(j)
        elif r.get("sky6_maybe_interior_as_sky", False):
            sky_px = sky_px | data.maybe_proxy_sky_interior(j, int(r.get("sky6_edge_px", 7)))
        if not r.get("sky_photo_splat_grad", True):
            C = sky_photo_stopgrad(C, rgb, alphas, sky_rgb, sky_px)
        photomask = (b & (LS.BIT_MOVING | LS.BIT_CAMFIXED | LS.BIT_PRIVACY)) == 0
        expo_on = bool(r.get("exposure_on", True)) and step >= r["exposure_start_frac"] * self.S
        if expo_on:
            # fit only where the render is reliable: splat-covered (alpha >= exposure_alpha_min), not sky / maybe sky,
            # unsaturated; at least exposure_min_share of the frame, else the photo keeps its previous estimate; smoothed
            # over visits (exposure_beta) and gauge-normalised in apply() (exposure.py GAUGE)
            fitmask = (photomask & ((b & (LS.BIT_SAT | LS.BIT_SKY | LS.BIT_SKY_MAYBE)) == 0)
                       & (al.detach() >= r["exposure_alpha_min"]))
            if r.get("exposure_model", "affine") == "gain_median":
                self.expo.solve_gain(C.detach(), img, fitmask, ti, beta=r["exposure_beta"],
                                     min_px=r["exposure_min_share"] * self.W * self.H)
            else:
                self.expo.solve(C.detach(), img, fitmask, ti, r["exposure_lam"], beta=r["exposure_beta"],
                                min_px=r["exposure_min_share"] * self.W * self.H)
            pred = self.expo.apply(C, ti)
        else:
            pred = C.clamp(0.0, 1.0)
        l_ph, l1, ss = LS.photometric(pred, img, photomask, self.win, r["clip_level"], r["w_l1"], r["w_dssim"])
        dmask = ((b & LS.BIT_VALID) != 0) & ((b & (LS.BIT_SKY | LS.BIT_SKY_MAYBE | LS.BIT_SAT | LS.BIT_EDGE | LS.BIT_MOVING)) == 0)
        l_d, n_d = LS.disparity_loss(ed, al, disp, dmask, self.fB, r["huber_delta_px"], r["depth_alpha_min"], z_min=self.near,
                                     w_near=r.get("w_near_depth", 1.0))
        pos, neg = LS.coverage_targets(b)
        neg = neg | sky_px                     # 2c: + the maybe-sky that SKY-6 counts as sky (when sky6_maybe_as_sky)
        l_b = LS.coverage_bce(al, pos, neg)
        wd = LS.depth_weight(step, self.S, r["w_depth0"], r["w_depth1"])
        # depth trust: no depth loss on a view whose static-world test rejects most of it (tensor op: no host sync)
        trust = (((b & LS.BIT_MOVING) != 0).float().mean() <= MAX_MOVING_SHARE_FOR_DEPTH).float()
        loss = l_ph + (wd * trust) * l_d + r["w_bce"] * l_b
        if timing:
            self.sync(); tt["forward"] = time.perf_counter() - t; t = time.perf_counter()
        loss.backward()
        if pose_d is not None:
            if pose_d.grad is None:
                raise RuntimeError("pose refinement: the renderer returned no gradient for the camera pose")
            self.poser.update(ti, pose_d.grad, step)
        g_op = self.params["opacities"].grad
        vis = (g_op.reshape(-1) != 0) if g_op is not None else torch.zeros(len(self.params["means"]), dtype=torch.bool, device=self.device)
        reg, parts = LS.regularisers(self.params, vis, self.L, r["w_opacity_reg"], r["w_scale_reg"], r["w_aniso"], r["aniso_max_ratio"])
        if reg.requires_grad:
            reg.backward()
        if timing:
            self.sync(); tt["backward"] = time.perf_counter() - t; t = time.perf_counter()
        lrm = self.lr_means(step)
        self.opts["means"].param_groups[0]["lr"] = lrm
        for n in PARAM_NAMES:
            self.opts[n].step(vis)
            self.opts[n].zero_grad(set_to_none=True)
        self.opt_sky.step()
        self.opt_sky.zero_grad(set_to_none=True)
        LS.clamp_scales_(self.params, r["max_scale_m"])
        self.angular_cap_(step)
        if timing:
            self.sync(); tt["optimiser"] = time.perf_counter() - t; t = time.perf_counter()
        self.strategy.step_post_backward(self.params, self.opts, self.sstate, step, info, lr=lrm)
        if timing:
            self.sync(); tt["strategy"] = time.perf_counter() - t
        self.step += 1
        return dict(loss=float(loss.detach()), l1=float(l1.detach()), ssim=float(ss.detach()), depth=float(l_d.detach()),
                    n_depth_px=n_d, bce=float(l_b.detach()), n_vis=int(vis.sum()), reg=parts, view=int(j), ti=int(ti),
                    expo_on=bool(expo_on), timing=tt)

    @torch.no_grad()
    def angular_cap_(self, step, force=False):
        """Phase 2c (eval 2026-10-22, work/_diag_eval/hide_by_footprint.json): no row may be wider than max_angular_px
        pixels (sigma x fx / d) as seen from its nearest training camera. The per-row cap is refreshed every
        angular_cap_every steps (and on demand after re-seeding); in between the cached cap is applied every step.
        max_angular_px <= 0 switches it off."""
        r = self.rec
        px = float(r.get("max_angular_px") or 0.0)
        if px <= 0:
            return
        n = len(self.params["means"])
        if force or getattr(self, "_ang_cap", None) is None or len(self._ang_cap) != n or step % int(r["angular_cap_every"]) == 0:
            if getattr(self, "_cam_centres", None) is None:
                self._cam_centres = torch.tensor(np.array([self.prep.T_wc(j)[:3, 3] for j in self.prep.train_ids]),
                                                 dtype=torch.float32, device=self.device)
            d = LS.nearest_cam_dist(self.params["means"].detach(), self._cam_centres)
            self._ang_cap = LS.angular_cap_log(d, self.Knp[0, 0], px)
        LS.clamp_scales_angular_(self.params, self._ang_cap)

    # ---------------------------------------------------------------- evaluation helpers
    @torch.no_grad()
    def render_view(self, j, data, with_sky=True, sh_deg=None):
        sh_deg = self.rec["sh_degree"] if sh_deg is None else sh_deg
        Tcw = data.T_cw[j]
        dirs = data.dirs(j, self.Knp, self.W, self.H) if with_sky else None
        ti = self._ti.get(j) if self.poser is not None else None
        if ti is not None and self.poser.active(self.step):           # training views at their refined pose
            Tcw = delta_inv_detached(self.poser.delta[ti]) @ Tcw
            if with_sky:
                Rw = data.R_wc[j]
                dirs = dirs @ (Rw @ self.poser.R_cam(ti) @ Rw.T).T
        colors, alphas, _ = self.render(self.params, Tcw, self.K, self.W, self.H, sh_deg, depth=True)
        rgb = colors[..., :3]
        if with_sky:
            rgb = SKY.composite(rgb, alphas, self.sky.sample(dirs))
        return rgb.clamp(0, 1), alphas[..., 0], colors[..., 3]

    @torch.no_grad()
    def probe(self, data):
        ps = []
        for j in self.probe_ids:
            rgb, _, _ = self.render_view(j, data)
            img = data.get(j)[0]
            mse = torch.mean((rgb - img) ** 2).clamp_min(1e-10)
            ps.append(float(-10 * torch.log10(mse)))
        return float(np.mean(ps)) if ps else float("nan")

    COV_EXCL = LS.BIT_SKY | LS.BIT_SKY_MAYBE | LS.BIT_SAT | LS.BIT_MOVING | LS.BIT_CAMFIXED

    @torch.no_grad()
    def coverage_pass(self, data, views, tag):
        out = []
        for j in views:
            _, al, _ = self.render_view(j, data, with_sky=False)
            b = data.get(j)[2]
            m = ((b & LS.BIT_VALID) != 0) & ((b & self.COV_EXCL) == 0)
            cov = float((al[m] >= self.rec["coverage_alpha"]).float().mean()) if bool(m.any()) else float("nan")
            out.append(dict(step=self.step, tag=tag, j=int(j), kf_id=int(self.prep.views[j]["kf_id"]), coverage=cov))
        self.coverage.extend(out)
        self._note_cov_median(out)
        return out

    def _note_cov_median(self, out):
        v = [c["coverage"] for c in out if np.isfinite(c["coverage"])]
        if v:
            med = float(np.median(v))
            self.k1["cov_median_last"] = med
            self.k1["cov_median_best"] = max(med, self.k1.get("cov_median_best", med))

    def reseed_due(self, s):
        """A coverage re-seed at every 0.05 S multiple below the refine stop (0.8 S), and never with less than 20 % of S
        still to go (lead 2026-10-18: the transient re-seed blobs must be trained away before the end)."""
        r = self.rec
        cov_every = self._every(r["coverage_every_frac"])
        return s % cov_every == 0 and s < int(r["refine_stop_frac"] * self.S) and (self.S - s) >= 0.2 * self.S

    def reseed_steps(self):
        return [s for s in range(1, self.S + 1) if self.reseed_due(s)]

    @torch.no_grad()
    def reseed(self, data):
        """Coverage re-seeding (5.2): holes (alpha < 0.5 on valid, non-sky, non-saturated, static RAFT pixels) are
        gathered from EVERY 5th training keyframe of the whole route, then subsampled uniformly (seeded by the step) to
        <= 2 % of cap_max (review S7: the phase-1 pass stopped at the first keyframes that filled the budget); dead /
        lowest-opacity rows move there."""
        r = self.rec
        max_n = int(r["reseed_max_frac"] * self.cap)
        st = int(r["reseed_stride_px"])
        fx, fy, cx, cy = self.Knp[0, 0], self.Knp[1, 1], self.Knp[0, 2], self.Knp[1, 2]
        pts, cols, logs, per_view = [], [], [], 0
        cand = self.prep.train_ids[::int(r["coverage_kf_stride"])]
        for j in cand:
            _, al, _ = self.render_view(j, data, with_sky=False)
            img, disp, b, _ = data.get(j)
            if float(((b & LS.BIT_MOVING) != 0).float().mean()) > MAX_MOVING_SHARE_FOR_DEPTH:
                continue                                    # untrusted RAFT (depth trust rule): never re-seed from it
            ok = ((b & LS.BIT_VALID) != 0) & ((b & (self.COV_EXCL | LS.BIT_EDGE)) == 0)
            hole = (ok & (al < r["coverage_alpha"]))[::st, ::st]
            if not bool(hole.any()):
                continue
            per_view += 1
            ii, jj = torch.nonzero(hole, as_tuple=True)
            v, u = ii * st, jj * st
            z = self.fB / disp[v, u].clamp_min(1e-6)
            X = torch.stack([(u.float() - cx) / fx * z, (v.float() - cy) / fy * z, z], -1)
            Twc_j = data.T_wc[j]
            if self.poser is not None and self.poser.active(self.step) and j in self._ti:
                Twc_j = self.poser.refined_T_wc(self._ti[j], Twc_j)
            T = torch.as_tensor(Twc_j, dtype=torch.float32, device=self.device)
            pts.append(X @ T[:3, :3].T + T[:3, 3])
            cols.append(img[v, u])
            logs.append(torch.log(reseed_scale(z, fx, st, self.fB / float(r["seed_min_disp_px"]), float(r["reseed_far_px"]))))
        if not pts:
            self.reseed_log.append(dict(step=self.step, moved=0, views_with_holes=0, candidates=0))
            return 0
        P, Cc, Lg = torch.cat(pts), torch.cat(cols), torch.cat(logs)
        n_cand = int(len(P))
        if n_cand > max_n:
            g = torch.Generator().manual_seed(int(r["seed"]) * 1_000_003 + int(self.step))
            sel = torch.randperm(n_cand, generator=g)[:max_n].to(P.device)
            P, Cc, Lg = P[sel], Cc[sel], Lg[sel]
        n = MW.reseed_rows(self.params, self.opts, P, Cc, Lg, max_n, r["min_opacity"])
        self.angular_cap_(self.step, force=True)
        self.reseed_log.append(dict(step=self.step, moved=int(n), views_with_holes=per_view, candidates=n_cand,
                                    views_scanned=len(cand)))
        return n

    # ---------------------------------------------------------------- kill rules
    def k1_check(self, rec_probe=None, loss500=None, cov_full=None):
        """K1 collapse (11.4): relocated dead share > 5 % of N for 3 refines; share of opacity < 0.05 up > 10 points;
        probe PSNR down >= 1 dB; 500-step loss up >= 30 %; any keyframe's coverage down >= 0.10.
        The switch k1_enabled exists for loop-mechanics unit tests on toy scenes only; production keeps it on."""
        if not self.rec.get("k1_enabled", True):
            return
        hist = self.strategy.history
        i0 = int(self.k1.get("reloc_checked_upto", 0))
        for i in range(max(i0, 2), len(hist)):
            win = hist[i - 2:i + 1]
            shares = [h["relocated"] / max(h["n_before"], 1) for h in win]
            if all(x > 0.05 for x in shares):
                self.k1["reloc_checked_upto"] = len(hist)
                raise KillRule("K1", "relocated dead share > 5 %% of N for 3 consecutive refines (%s at steps %s)"
                               % ([round(x, 3) for x in shares], [h["step"] for h in win]))
        self.k1["reloc_checked_upto"] = len(hist)
        if rec_probe is not None and np.isfinite(rec_probe):
            best = self.k1.get("best_probe")
            if best is not None and rec_probe <= best - 1.0:
                raise KillRule("K1", "probe PSNR %.2f dB is >= 1 dB below its best %.2f" % (rec_probe, best))
            why = self.low_alpha_collapse(rec_probe, best)
            if why:
                raise KillRule("K1", why)
            self.k1["best_probe"] = rec_probe if best is None else max(best, rec_probe)
            a = float((torch.sigmoid(self.params["opacities"].detach()) < 0.05).float().mean())
            amin = self.k1.get("low_alpha_min")
            self.k1["low_alpha_min"] = a if amin is None else min(amin, a)
            self.k1["low_alpha_last"] = a
        if rec_probe is not None and np.isfinite(rec_probe):
            self.k1["probe_last"] = float(rec_probe)
        if loss500 is not None:
            prev = self.k1.get("loss500_prev")
            if prev is not None and loss500 >= K1_LOSS500_RISE * prev and self.step > 2 * self.n_train:
                corr = self.k1_corroboration()
                if corr:
                    raise KillRule("K1", "500-step loss rose >= 30 %% (%.4f vs %.4f) AND %s" % (loss500, prev, corr))
                self.log("K1 500-step loss rose >= 30 %% (%.4f vs %.4f) but is NOT corroborated (probe last %s best %s, "
                         "coverage median last %s best %s): not a collapse (amended rule, THRESHOLDS 10.9)"
                         % (loss500, prev, self.k1.get("probe_last"), self.k1.get("best_probe"),
                            self.k1.get("cov_median_last"), self.k1.get("cov_median_best")))
            self.k1["loss500_prev"] = loss500
        if cov_full is not None:
            prevc = self.k1.get("cov_full_prev") or {}
            for c in cov_full:
                p = prevc.get(str(c["j"]))
                if p is not None and np.isfinite(c["coverage"]) and c["coverage"] <= p - 0.10:
                    raise KillRule("K1", "keyframe %d coverage fell by >= 0.10 (%.3f -> %.3f)" % (c["kf_id"], p, c["coverage"]))
            self.k1["cov_full_prev"] = {str(c["j"]): c["coverage"] for c in cov_full if np.isfinite(c["coverage"])}

    def k1_corroboration(self):
        """'' or why the state corroborates a collapse: probe >= 0.5 dB below its best, or the coverage median >= 0.05
        below its best (the amended low-alpha rule's corroboration, reused for the 500-step-loss sub-rule)."""
        pl, pb = self.k1.get("probe_last"), self.k1.get("best_probe")
        if pl is not None and pb is not None and pl <= pb - K1_LOW_ALPHA_PROBE_DB:
            return "probe %.2f dB >= %.1f dB below its best %.2f" % (pl, K1_LOW_ALPHA_PROBE_DB, pb)
        cl, cb = self.k1.get("cov_median_last"), self.k1.get("cov_median_best")
        if cl is not None and cb is not None and cl <= cb - K1_LOW_ALPHA_COV:
            return "coverage median %.4f >= %.2f below its best %.4f" % (cl, K1_LOW_ALPHA_COV, cb)
        return ""

    def low_alpha_collapse(self, rec_probe, best):
        """K1 low-alpha sub-rule, AMENDED (lead 2026-10-19, evidence work/_diag_k1b): a rise of the opacity < 0.05 share by
        > 10 points counts as collapse only when CORROBORATED by the probe down >= 0.5 dB from its best OR the median
        keyframe coverage down >= 0.05 from its best. (The killed pilot at 21630: share 0.004 -> 0.130 while the probe rose
        20.97 -> 22.09 dB and the coverage median stayed 0.997-0.998; hiding all 262,518 such rows changed the probe by
        -0.012 dB: MCMC + the opacity regulariser shedding redundant splats.) -> reason string or None."""
        a = float((torch.sigmoid(self.params["opacities"].detach()) < 0.05).float().mean())
        amin = self.k1.get("low_alpha_min")
        if amin is None or a <= amin + K1_LOW_ALPHA_RISE:
            return None
        probe_down = best is not None and rec_probe is not None and rec_probe <= best - K1_LOW_ALPHA_PROBE_DB
        cb, cl = self.k1.get("cov_median_best"), self.k1.get("cov_median_last")
        cov_down = cb is not None and cl is not None and cl <= cb - K1_LOW_ALPHA_COV
        if not (probe_down or cov_down):
            if not self.k1.get("low_alpha_logged"):
                self.log("K1 low-alpha share %.3f (min %.3f) rose > 10 points but is NOT corroborated (probe %.2f vs best %s, "
                         "coverage median %s vs best %s): not a collapse (amended rule)" % (a, amin, rec_probe, best, cl, cb))
                self.k1["low_alpha_logged"] = True
            return None
        return ("share of opacity < 0.05 rose by > 10 points (%.3f vs min %.3f) AND %s" % (
            a, amin, "probe %.2f dB >= %.1f dB below its best %.2f" % (rec_probe, K1_LOW_ALPHA_PROBE_DB, best) if probe_down
            else "median keyframe coverage %.3f >= %.2f below its best %.3f" % (cl, K1_LOW_ALPHA_COV, cb)))

    def k1_reevaluate_override(self, k, ov):
        """At resume after a K1 kill covered by a lead override file: re-evaluate the AMENDED K1 on the restored state
        (and K2, if the kill pre-empted it at 25 % of S). Continue only if both pass; the kill record is kept (renamed)."""
        if not self.k1.get("cov_median_best"):
            by = {}
            for c in self.coverage:
                if np.isfinite(c["coverage"]):
                    by.setdefault(c["step"], []).append(c["coverage"])
            meds = [float(np.median(by[x])) for x in sorted(by)]
            if meds:
                self.k1["cov_median_best"], self.k1["cov_median_last"] = max(meds), meds[-1]
        probes = [(c["step"], c["probe"]) for c in self.curve if c.get("probe") is not None]
        last = probes[-1][1] if probes else None
        prev_best = max((p for st, p in probes[:-1]), default=None)
        why = self.low_alpha_collapse(last, prev_best)
        if not why and "500-step loss" in str(k.get("detail")):
            self.k1["probe_last"] = last if last is not None else self.k1.get("probe_last")
            c = self.k1_corroboration()
            why = ("500-step loss kill corroborated: %s" % c) if c else ""
        if why or (last is not None and prev_best is not None and last <= prev_best - 1.0):
            raise SystemExit("K1 override %s: the amended K1 still fires on the restored state: %s" % (ov.get("_path"), why or "probe down >= 1 dB"))
        k2 = None
        if self.k2_db is not None and int(k.get("step", -1)) == self._every(0.25):
            at = [p for st, p in probes if st == int(k["step"])]
            k2 = dict(step=int(k["step"]), probe=at[-1] if at else None, bar=float(self.k2_db))
            if not at or at[-1] < float(self.k2_db):
                raise SystemExit("K1 override: K2 at 25 %% of S fails on the restored state: %s" % k2)
        kp = os.path.join(self.work, "kill.json")
        dst = os.path.join(self.work, "kill_overridden_%s_%s.json" % (k.get("rule"), time.strftime("%Y%m%d%H%M%S")))
        U._replace_retry(kp, dst)
        rec = dict(override=ov.get("_path"), run_id=self.run_id, kill=k, kill_record_kept_as=dst, amended_rule=ov.get("amended_rule"),
                   restored=dict(step=self.step, probe_last=last, probe_best_before=prev_best,
                                 low_alpha_share=float((torch.sigmoid(self.params["opacities"].detach()) < 0.05).float().mean()),
                                 low_alpha_min=self.k1.get("low_alpha_min"), cov_median_last=self.k1.get("cov_median_last"),
                                 cov_median_best=self.k1.get("cov_median_best")), k2=k2, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        U.atomic_json(os.path.join(self.work, "k1_override_applied.json"), rec)
        self.log("K1 kill at step %s OVERRIDDEN by %s: the amended K1 passes on the restored state %s; K2 %s; continuing run %s"
                 % (k.get("step"), ov.get("_path"), rec["restored"], k2, self.run_id))
        self.k1["best_probe"] = max(p for _, p in probes) if probes else self.k1.get("best_probe")

    def k2_check(self, s, probe_db):
        """K2 (review S5): at 25 % of S the probe must be >= the row's registered bar (THRESHOLDS.md; rosario rows 18.84 dB)."""
        if self.k2_db is None or probe_db is None or s != self._every(0.25):
            return
        if probe_db < float(self.k2_db):
            raise KillRule("K2", "probe %.2f dB at 25 %% of S (step %d) < the registered bar %.2f dB" % (probe_db, s, float(self.k2_db)))

    def k5_check(self, step_time_s):
        if self.device.type != "cuda":
            return
        res = torch.cuda.memory_reserved(self.device) / 2 ** 30
        if res > self.rec["vram_guard_gib"]:
            if self.mem_frozen_at is None:
                self.strategy.cap_max = len(self.params["means"])
                self.mem_frozen_at = dict(step=self.step, n=len(self.params["means"]), reserved_gib=res)
                self.log("K5: reserved %.2f GiB > %.1f GiB guard: N frozen at %d" % (res, self.rec["vram_guard_gib"], self.strategy.cap_max))
            if self.probe_step_s and step_time_s > 2 * self.probe_step_s:
                raise KillRule("K5", "memory guard AND step time %.1f ms > 2 x the 0d probe %.1f ms" % (1e3 * step_time_s, 1e3 * self.probe_step_s))

    def k6_check(self, wall_s):
        if self.estimate_s is None:
            return
        frac = self.step / self.S
        if frac >= self.rec["k6_check_frac"] and not getattr(self, "_k6_done", False):
            self._k6_done = True
            proj = wall_s / frac
            if proj > self.rec["k6_factor"] * self.estimate_s:
                raise KillRule("K6", "projected %.2f h > %.1f x the 0d estimate %.2f h" % (proj / 3600, self.rec["k6_factor"], self.estimate_s / 3600))

    # ---------------------------------------------------------------- status (review S1: never kills the run)
    def heartbeat(self, extra):
        try:
            st = dict(row=self.prep.row, stage=self.stage_name, pid=os.getpid(), run_id=self.run_id, step=self.step,
                      total_steps=self.S, n_gaussians=len(self.params["means"]), time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                      **self.vram(), **extra)
            U.atomic_json(os.path.join(self.work, "status.json"), st)
        except OSError as e:
            self.log("status.json write failed (ignored, retried at the next heartbeat): %r" % e)

    def write_curve(self):
        try:
            lines = ["step,loss,l1,ssim,depth,bce,n_gauss,n_vis,probe_psnr,seconds"]
            for c in self.curve:
                lines.append("%d,%.5f,%.5f,%.4f,%.5f,%.5f,%d,%d,%s,%.1f" % (
                    c["step"], c["loss"], c["l1"], c["ssim"], c["depth"], c["bce"], c["n"], c["n_vis"],
                    "" if c.get("probe") is None else "%.3f" % c["probe"], c["t"]))
            U.atomic_write_text(os.path.join(self.work, "curve.csv"), "\n".join(lines) + "\n")
            rows = ["step,tag,j,kf_id,coverage"] + ["%d,%s,%d,%d,%s" % (c["step"], c["tag"], c["j"], c["kf_id"],
                                                                         "" if not np.isfinite(c["coverage"]) else "%.4f" % c["coverage"])
                                                   for c in self.coverage]
            U.atomic_write_text(os.path.join(self.work, "coverage.csv"), "\n".join(rows) + "\n")
        except OSError as e:
            self.log("curve / coverage write failed (ignored, retried at the next write): %r" % e)

    # ---------------------------------------------------------------- main loop
    def run(self, data, resume=True, stop_after=None):
        """Train to S steps (or stop after `stop_after` steps of this session: the planned-kill test).
        resume=False starts a NEW run (new run id, the previous run's checkpoints and logs cleared first)."""
        r = self.rec
        resumed = self.try_resume() if resume else None
        if resumed is None:
            self.start_fresh_run()
            self.setup_fresh()
            self.log("fresh run %s: %d training views, %d held out (%d probe), %d seeds, cap %d, S %d, L %.2f m, noise_lr %.3g, "
                     "renderer %s, device %s, inputs %s" % (self.run_id, self.n_train, len(self.prep.hold_ids), len(self.probe_ids),
                                                            len(self.params["means"]), self.cap, self.S, self.L, self.strategy.noise_lr,
                                                            getattr(self.render, "kind", "?"), self.device, self.input_digest[:12]))
        self.t_session0 = time.time()
        live = None
        if self.want_live:
            hold_j = self.prep.hold_ids[len(self.prep.hold_ids) // 2] if self.prep.hold_ids else self.prep.train_ids[0]
            Tov, Kov, Wov, Hov = overview_camera([data.T_wc[j][:3, 3] for j in self.prep.train_ids], self.prep.up, self.Knp)
            live = LivePreview(os.path.join(self.work, "live"), self.prep.row, self.stage_name, self.render, self.Knp, self.W, self.H,
                               dict(j=hold_j, kf_id=self.prep.views[hold_j]["kf_id"], img=data.img_numpy(hold_j),
                                    T_cw=IO.inv_T(data.T_wc[hold_j])), (Tov, Kov, Wov, Hov),
                               r["live_every_init"], r["live_every_min"], r["live_every_max"], r["live_overhead_max"],
                               r["live_width"], log=self.log, device=self.device)
        cov_every = self._every(r["coverage_every_frac"])
        stop_cov = int(r["refine_stop_frac"] * self.S)
        full_at = sorted(set(int(round(f * self.S)) for f in r["coverage_full_at"]))
        ms_at = {int(round(f * self.S)): f for f in self.milestones}
        last_ck = time.time()
        last_hb = 0.0
        t_steps, n_steps = 0.0, 0
        session_steps = 0
        last_probe = self.curve[-1].get("probe") if self.curve else None
        try:
            while self.step < self.S:
                timing = (self.step + 1) % 500 == 0
                t0 = time.perf_counter()
                out = self.train_step(data, timing=timing)
                dt = time.perf_counter() - t0
                t_steps += dt
                n_steps += 1
                session_steps += 1
                self.loss_win.append(out["loss"])
                s = self.step                           # steps done
                if not np.isfinite(out["loss"]):
                    raise KillRule("K1", "non-finite loss at step %d" % s)
                if timing:
                    self.telemetry.append(dict(step=s, **{k: round(v * 1e3, 3) for k, v in out["timing"].items()}, **self.vram()))
                if s in ms_at:                           # review N5c: the milestone is written before any K1 check
                    self._milestone(ms_at[s])
                # ORDER (lead 2026-10-18, diagnosis work/_diag_k1): probe -> re-seed -> coverage. The K1 / K2 probe is the
                # state BEFORE this step's re-seed: freshly re-seeded far-field splats (opacity 0.1, footprint-sized) bleed
                # into the sky for a few hundred steps and depressed the killed pilot's probe by 2.2 dB at step 8652.
                probe_now = (s % cov_every == 0) or s == self.S or s == self._every(0.25)
                probe_val = None
                if probe_now and self.probe_ids:
                    probe_val = self.probe(data)
                    self.events.append(("probe", s))
                if self.coverage_on and self.reseed_due(s):
                    n_moved = self.reseed(data)
                    self.events.append(("reseed", s))
                    self.log("coverage re-seed at %d: %d rows moved (%s)" % (s, n_moved, self.reseed_log[-1]))
                if self.coverage_on and s in full_at:
                    cf = self.coverage_pass(data, sorted(set(self.prep.train_ids) | set(self.prep.hold_ids)), "full")
                    self.events.append(("coverage", s))
                    self.k1_check(cov_full=cf)
                elif self.coverage_on and s % cov_every == 0:
                    self.coverage_pass(data, self.prep.train_ids[::int(r["coverage_kf_stride"])], "every5th")
                    self.events.append(("coverage", s))
                if s % 500 == 0 or probe_now:
                    rec = dict(step=s, loss=out["loss"], l1=out["l1"], ssim=out["ssim"], depth=out["depth"], bce=out["bce"],
                               n=len(self.params["means"]), n_vis=out["n_vis"], t=self.wall_prev + time.time() - self.t_session0)
                    if probe_val is not None:
                        rec["probe"] = probe_val
                        last_probe = rec["probe"]
                    self.curve.append(rec)
                    l5 = None
                    if s % 500 == 0:
                        l5 = float(np.mean(self.loss_win)) if self.loss_win else None
                        self.loss_win = []
                    self.k1_check(rec_probe=rec.get("probe"), loss500=l5)
                    self.k2_check(s, rec.get("probe"))
                    self.write_curve()
                    sps = t_steps / max(n_steps, 1)
                    self.log("step %d/%d loss %.4f L1 %.4f SSIM %.3f disp %.4f N %d vis %d %.1f ms/step%s"
                             % (s, self.S, out["loss"], out["l1"], out["ssim"], out["depth"], rec["n"], out["n_vis"], 1e3 * sps,
                                ("  probe(%d) %.2f dB" % (len(self.probe_ids), rec["probe"])) if "probe" in rec else ""))
                    self.k5_check(sps)
                wall = self.wall_prev + time.time() - self.t_session0
                self.k6_check(wall)
                if live is not None and live.due(s):
                    sps = t_steps / max(n_steps, 1)
                    lat = out["view"]
                    live.write(s, self.S, dict(n_gaussians=len(self.params["means"]), loss=out["loss"], heldout_probe_psnr=last_probe,
                                               it_per_s=(1.0 / sps if sps > 0 else None), eta_s=(self.S - s) * sps, run_id=self.run_id,
                                               **self.vram()),
                               self.params, min(s // r["sh_ramp_steps"], r["sh_degree"]), self.sky,
                               dict(j=lat, kf_id=self.prep.views[lat]["kf_id"], img=data.img_numpy(lat), T_cw=IO.inv_T(data.T_wc[lat])),
                               sps)
                if time.time() - last_ck >= self.ckpt_every_s and s < self.S:
                    self._rolling()
                    last_ck = time.time()
                if time.time() - last_hb >= self.rec["heartbeat_s"]:
                    sps = t_steps / max(n_steps, 1)
                    self.heartbeat(dict(loss=out["loss"], probe=last_probe, ms_per_step=1e3 * sps, eta_s=(self.S - s) * sps))
                    last_hb = time.time()
                if s % 100 == 0 and os.path.exists(os.path.join(self.work, "STOP")):
                    self._rolling()
                    self.log("STOP file seen at step %d: checkpoint written, stopping cleanly (resume with --resume)" % s)
                    return dict(stopped=True, step=self.step, reason="STOP file")
                # optional chunk limit (max_chunk_s > 0): one process trains at most max_chunk_s, then checkpoints and
                # stops; `--resume` continues. xclean-maps default 0 = off (the internal driver used 1800 s chunks and
                # quiet-machine flags; neither is shipped)
                if s % 100 == 0 and s < self.S:
                    chunk = float(self.rec.get("max_chunk_s") or 0)
                    if chunk > 0 and time.time() - self.t_session0 >= chunk:
                        self._rolling()
                        why = "chunk limit %.0f s" % chunk
                        self.log("%s at step %d: checkpoint written, stopping cleanly (resume with --resume)" % (why, s))
                        return dict(stopped=True, step=self.step, reason=why)
                if stop_after is not None and session_steps >= stop_after:
                    self._rolling()
                    self.log("planned stop after %d session steps at step %d (checkpoint written)" % (session_steps, s))
                    return dict(stopped=True, step=self.step)
        except KillRule as k:
            p = self._rolling()
            U.atomic_json(os.path.join(self.work, "kill.json"), dict(rule=k.rule, detail=k.detail, step=self.step, checkpoint=p,
                                                                      run_id=self.run_id, time=time.strftime("%Y-%m-%dT%H:%M:%S")))
            self.log("KILL %s at step %d: %s (checkpoint %s; stopped for review)" % (k.rule, self.step, k.detail, p))
            raise
        self.write_curve()
        wall = self.wall_prev + time.time() - self.t_session0
        final = os.path.join(self.work, "final_params.pt")
        U.assert_writable(final)
        torch.save(self.checkpoint_blob(), final + ".tmp")
        _fsync_replace(final + ".tmp", final)        # review N3
        self.heartbeat(dict(done=True, wall_s=wall))
        info = dict(run_id=self.run_id, config_hash=self.config_hash, input_digest=self.input_digest, steps=self.S,
                    n_gaussians=len(self.params["means"]), wall_s=wall, ms_per_step=1e3 * t_steps / max(n_steps, 1),
                    vram=self.vram(), relocations=len(self.strategy.history), reseeds=self.reseed_log,
                    live_previews=(live.n_written if live else 0), live_failed=(live.n_failed if live else 0),
                    live_every=(live.every if live else None), live_cost_s=(live.cost_s if live else None),
                    live_overhead_frac=(live.overhead_frac if live else None), final_params=final, mem_frozen_at=self.mem_frozen_at,
                    final_params_sha1=U.sha1_file(final))
        if self.poser is not None:
            rep_ = self.poser.report([self.prep.views[j]["kf_id"] for j in self.prep.train_ids], self.t_train)
            U.atomic_json(os.path.join(self.work, "pose_refine.json"), dict(rep_, run_id=self.run_id, rule=PR.__doc__))
            info["pose_refine"] = {k: v for k, v in rep_.items() if k != "per_kf"}
            self.log("pose refinement: rot median %.3f / max %.3f deg, trans median %.4f / max %.4f m, cap hits %d"
                     % (rep_["rot_deg_median"], rep_["rot_deg_max"], rep_["trans_m_median"], rep_["trans_m_max"], rep_["n_cap_hits"]))
        U.atomic_json(os.path.join(self.work, "train_summary.json"), info)
        self.log("DONE %d steps, %d gaussians, %.0f s" % (self.S, info["n_gaussians"], wall))
        return info
