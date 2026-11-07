# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Readers of xclean's own outputs (the camera path of a map). Read-only.

Two layouts are read (the release's own writers, read in their source, FRAME RULE):

1. A `xclean replay` / `xclean live` run folder (xclean/live/harness.py, xclean/vio/engine_live.py):
     recorder.jsonl      one JSON message per published pose; "T" = the row-major (C-order) 4x4
                         T_world_cam0_rectified (live/protocol.make_msg: np.asarray(T).reshape(-1)); streams LIVE-ODOM and
                         LIVE-MAP; ends with {"recorder_end": true}
     trajectory_live.csv fallback (live/oak_demo.write_trajectory_live): t,x,y,z,qx,qy,qz,qw,frame_index,state,...
                         quaternion = scipy as_quat (x, y, z, w), W LAST; header names T_world_cam0_rectified
     p1_frames.jsonl     one record per frame; "kf": true on the engine's keyframes
     engine_summary.json p1.g_w (gravity in w) and p1.vi (IMU admitted)
     harness.json        "sequence_folder" (the sequence the run was made from)
   This is the path the website maps were built from (internal builder sources.P1Run, kept verbatim below).
   Views = the engine's keyframes (p1_frames.jsonl kf=true) + the first published frame. Poses = the LIVE stream
   (LIVE-MAP when it carries a correction, else LIVE-ODOM), unchanged.

2. A `xclean run` output folder (xclean/run.py + xclean/kit/writer.py):
     kit/trajectory_kf.csv      "# t,x,y,z,qx,qy,qz,qw,kf_id,frame_index" (W LAST), T_world_cam0_rectified, the FINAL
                                optimised keyframe poses (after loop closure and global BA)
     kit/trajectory_frames.csv  per-frame poses re-derived from the final trajectory
     kit/run_manifest.json      "dataset" (the sequence folder) and "frame_convention"
     summary.json               the run summary
   Views = the kit's keyframes (every row of kit/trajectory_kf.csv). Poses = those final keyframe poses.
   NOTE: the website maps were NOT built from this layout; it is supported because it is what `xclean run` writes.

Up (for the sky rule and the previews): -g_w of the engine when the IMU was admitted; else the accelerometer mean
over the first 1 s of the run (sources.accel_up, checked); else the mean camera -y.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

FRAME_CONVENTION = "T_world_cam0_rectified"


def _sha1_file(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rot_ok(R, tol=1e-5):
    return bool(np.abs(R.T @ R - np.eye(3)).max() < tol and abs(np.linalg.det(R) - 1.0) < tol)


def detect_layout(run_dir):
    """'replay' (recorder.jsonl / trajectory_live.csv) or 'run' (kit/trajectory_kf.csv); None when neither."""
    if os.path.isfile(os.path.join(run_dir, "recorder.jsonl")) or os.path.isfile(os.path.join(run_dir, "trajectory_live.csv")):
        return "replay"
    if os.path.isfile(os.path.join(run_dir, "kit", "trajectory_kf.csv")):
        return "run"
    return None


def open_run(run_dir):
    lay = detect_layout(run_dir)
    if lay == "replay":
        return ReplayRun(run_dir)
    if lay == "run":
        return KitRun(run_dir)
    raise SystemExit("%s is not an xclean output folder: no recorder.jsonl, trajectory_live.csv or kit/trajectory_kf.csv"
                     % run_dir)


class _RunBase:
    layout = None

    def _scan_files(self, names):
        self.files = {}
        for fn in names:
            p = os.path.join(self.dir, fn)
            if os.path.isfile(p):
                self.files[fn] = dict(path=p.replace("\\", "/"), bytes=os.path.getsize(p), sha1=_sha1_file(p))

    def _rj(self, fn):
        if fn not in self.files:
            return None
        with open(self.files[fn]["path"], encoding="utf-8") as fh:
            return json.load(fh)

    def coasted_pct(self):
        n = len(self.frames)
        c = sum(1 for f in self.frames if self.state[f] not in ("tracking", "relocalized"))
        return 100.0 * c / max(n, 1)

    def gravity(self):
        """(g_w, rule) from the run's own files, or (None, why)."""
        return None, "no gravity estimate in the run files"

    def up(self, ds=None):
        g, rule = self.gravity()
        if g is not None:
            u = -np.asarray(g, float)
            return u / np.linalg.norm(u), rule
        if ds is not None:
            from .sources import accel_up
            r = accel_up(ds, self)
            if r is not None:
                return r
        Rs = [self.T[f][:3, :3] for f in self.frames]
        u = np.mean([R @ np.array([0.0, -1.0, 0.0]) for R in Rs], 0)
        return u / np.linalg.norm(u), "mean camera -y over the run (no gravity estimate)"

    def no_gt_checks(self):
        """Checks that need no ground truth (reported in plan.json; a build is refused only when a check says the run
        cannot be used: no poses, or no keyframes)."""
        c = dict(layout=self.layout, n_frames_with_pose=len(self.frames), n_views=len(self.kf_frames or []),
                 coasted_pct=round(self.coasted_pct(), 3), **self.checks)
        if self.frames:
            P = np.array([self.T[f][:3, 3] for f in self.frames])
            c["path_length_m"] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
            t = np.array([self.t[f] for f in self.frames])
            c["time_monotonic"] = bool(np.all(np.diff(t) > 0)) if len(t) > 1 else True
            c["duration_s"] = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        return c

    def write_tum(self, path_kf, path_frames, kf_frames):
        """The poses used, as CSV files (t,x,y,z,qx,qy,qz,qw,...; W LAST)."""
        from scipy.spatial.transform import Rotation as Rot
        from . import util as U
        lines = ["# xclean %s keyframes (%s; q = x,y,z,w W LAST); run %s" % (self.stream, FRAME_CONVENTION, self.dir),
                 "# t,x,y,z,qx,qy,qz,qw,kf_id,frame_index"]
        for f in kf_frames:
            T = self.T[f]
            q = Rot.from_matrix(T[:3, :3]).as_quat()
            lines.append("%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%d,%d" % (self.t[f], *T[:3, 3], *q, f, f))
        U.atomic_write_text(path_kf, "\n".join(lines) + "\n")
        lines = ["# xclean %s frames (%s; W LAST); run %s" % (self.stream, FRAME_CONVENTION, self.dir),
                 "# t,x,y,z,qx,qy,qz,qw,frame_index,state"]
        for f in self.frames:
            T = self.T[f]
            q = Rot.from_matrix(T[:3, :3]).as_quat()
            lines.append("%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f,%d,%s" % (self.t[f], *T[:3, 3], *q, f, self.state[f]))
        U.atomic_write_text(path_frames, "\n".join(lines) + "\n")

    @property
    def pose_file(self):
        for fn in ("recorder.jsonl", "trajectory_live.csv", "kit/trajectory_kf.csv"):
            if fn in self.files:
                return self.files[fn]
        return None


class ReplayRun(_RunBase):
    """A `xclean replay` / `xclean live` run folder. The reader is the internal builder's P1Run, unchanged in its maths."""
    layout = "replay"

    def __init__(self, run_dir):
        self.dir = run_dir.replace("\\", "/").rstrip("/")
        if not os.path.isdir(self.dir):
            raise SystemExit("run folder not found: %s" % self.dir)
        self._scan_files(("recorder.jsonl", "trajectory_live.csv", "p1_frames.jsonl", "engine_summary.json",
                          "harness.json", "live_score.json", "feeder.json", "demo.json", "summary.json"))
        self.summary = self._rj("engine_summary.json") or self._rj("summary.json")
        self.harness = self._rj("harness.json")
        self.checks = {}
        if "recorder.jsonl" in self.files:
            self._from_recorder()
        elif "trajectory_live.csv" in self.files:
            self._from_csv()
        self.kf_frames = self._kfs()

    def sequence_folder(self):
        """The sequence folder the run was made from, as the run recorded it (or None)."""
        h = self.harness or {}
        for k in ("sequence_folder", "dataset"):
            if h.get(k):
                return h[k]
        f = self._rj("feeder.json") or {}
        if isinstance(f.get("dataset"), str):
            return f["dataset"]
        return None

    # ---- readers (internal builder sources.P1Run, unchanged)
    def _from_recorder(self):
        msgs = {"LIVE-ODOM": {}, "LIVE-MAP": {}}
        invalid, end, n_other, corr_fields = [], None, 0, set()
        with open(self.files["recorder.jsonl"]["path"], encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                m = json.loads(ln)
                if "invalid" in m:
                    invalid.append(m)
                    continue
                if m.get("recorder_end"):
                    end = m
                    continue
                s = m.get("stream")
                if s not in msgs:
                    n_other += 1
                    continue
                fi = int(m["frame_index"])
                if fi in msgs[s]:
                    invalid.append(dict(dup=(fi, s)))
                msgs[s][fi] = m
                if s == "LIVE-MAP":
                    for k in m:
                        if k in ("T_map_odom", "correction", "map_correction", "T_corr", "loop_correction"):
                            corr_fields.add(k)
        odo, mp = msgs["LIVE-ODOM"], msgs["LIVE-MAP"]
        both = sorted(set(odo) & set(mp))
        dmax = 0.0
        for fi in both:
            a = np.asarray(odo[fi]["T"], float)
            b = np.asarray(mp[fi]["T"], float)
            if np.isfinite(a).all() and np.isfinite(b).all():
                dmax = max(dmax, float(np.abs(a - b).max()))
        corrected = bool(corr_fields) or dmax > 1e-12
        self.stream = "LIVE-MAP" if (corrected and mp) else ("LIVE-ODOM" if odo else "LIVE-MAP")
        src = mp if self.stream == "LIVE-MAP" else odo
        self.t, self.T, self.state, self.n_inliers = {}, {}, {}, {}
        n_nonfinite, n_badrot = 0, 0
        for fi in sorted(src):
            m = src[fi]
            T = np.asarray(m["T"], float).reshape(4, 4)        # row-major (protocol.make_msg reshape(-1))
            if not np.isfinite(T).all():
                n_nonfinite += 1
                continue
            if not _rot_ok(T[:3, :3], 1e-4) or np.abs(T[3] - [0, 0, 0, 1]).max() > 1e-9:
                n_badrot += 1
                continue
            self.t[fi], self.T[fi], self.state[fi] = float(m["t_pose"]), T, str(m.get("state"))
            self.n_inliers[fi] = m.get("n_inliers")
        self.frames = sorted(self.T)
        self.checks.update(source="recorder.jsonl", stream=self.stream, stream_rule=(
            "LIVE-MAP if it carries a correction (field %s or a pose differing from LIVE-ODOM), else LIVE-ODOM"
            % sorted(corr_fields)),
            n_live_odom=len(odo), n_live_map=len(mp), map_vs_odom_max_abs=dmax, correction_fields=sorted(corr_fields),
            invalid_records=len(invalid), recorder_end=bool(end), duplicates_reported=(end or {}).get("duplicates"),
            n_nonfinite=n_nonfinite, n_bad_rotation=n_badrot, other_messages=n_other)

    def _from_csv(self):
        from scipy.spatial.transform import Rotation as Rot
        self.stream = "trajectory_live.csv (LIVE-ODOM as written by its producer)"
        self.t, self.T, self.state, self.n_inliers = {}, {}, {}, {}
        hdr = []
        with open(self.files["trajectory_live.csv"]["path"], encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("#"):
                    hdr.append(ln.strip())
                    continue
                c = ln.strip().split(",")
                if len(c) < 10:
                    continue
                T = np.eye(4)
                T[:3, :3] = Rot.from_quat([float(x) for x in c[4:8]]).as_matrix()     # scipy (x, y, z, w): W LAST
                T[:3, 3] = [float(x) for x in c[1:4]]
                fi = int(c[8])
                self.t[fi], self.T[fi], self.state[fi] = float(c[0]), T, c[9]
        if not any(FRAME_CONVENTION in h for h in hdr):
            raise SystemExit("%s: header does not state %s: %s" % (self.files["trajectory_live.csv"]["path"],
                                                                    FRAME_CONVENTION, hdr))
        self.frames = sorted(self.T)
        self.checks.update(source="trajectory_live.csv", header=hdr, quaternion="scipy as_quat (x, y, z, w), W LAST")

    def _kfs(self):
        if "p1_frames.jsonl" not in self.files:
            return None
        kf = []
        with open(self.files["p1_frames.jsonl"]["path"], encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                d = json.loads(ln)
                if d.get("kf"):
                    kf.append(int(d["frame"]))
        if self.frames:
            kf = sorted(set(kf) | {self.frames[0]})
        missing = [k for k in kf if k not in self.T]
        self.checks["kf_rule"] = "the engine's own keyframes: p1_frames.jsonl kf=true + the first published frame"
        self.checks["kf_without_pose"] = missing[:20]
        return [k for k in kf if k in self.T]

    @property
    def p1(self):
        return (self.summary or {}).get("p1") or {}

    def gravity(self):
        g = self.p1.get("g_w")
        if g is not None and self.p1.get("vi"):
            return g, "-g_w of the engine's final window state (engine_summary.json p1.g_w; IMU admitted)"
        return None, "the engine did not admit the IMU (no g_w)"

    def engine_identity(self):
        h = self.harness or {}
        return dict(engine_md5=h.get("engine_md5") or (self.summary or {}).get("engine_md5"), mode=h.get("mode"),
                    label=h.get("label"), valid=h.get("valid"), tag=h.get("tag"))


class KitRun(_RunBase):
    """A `xclean run` output folder: the map build kit (kit/trajectory_kf.csv, kit/trajectory_frames.csv)."""
    layout = "run"

    def __init__(self, run_dir):
        self.dir = run_dir.replace("\\", "/").rstrip("/")
        self._scan_files(("kit/trajectory_kf.csv", "kit/trajectory_frames.csv", "kit/run_manifest.json", "summary.json",
                          "kit/calib.json"))
        self.summary = self._rj("summary.json")
        self.manifest = self._rj("kit/run_manifest.json") or {}
        conv = self.manifest.get("frame_convention")
        if conv and conv != FRAME_CONVENTION:
            raise SystemExit("%s: frame convention %r, expected %s" % (self.dir, conv, FRAME_CONVENTION))
        self.stream = "xclean run: final keyframe poses (kit/trajectory_kf.csv)"
        self.t, self.T, self.state = {}, {}, {}
        kf_t, kf_T, kfs = {}, {}, []
        hdr = []
        with open(self.files["kit/trajectory_kf.csv"]["path"], encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("#"):
                    hdr.append(ln.strip())
                    continue
                c = ln.strip().split(",")
                if len(c) < 10:
                    continue
                T = _T_from_row(c)
                fi = int(c[9])
                kf_t[fi], kf_T[fi] = float(c[0]), T
                kfs.append(fi)
        if not any(FRAME_CONVENTION in h for h in hdr):
            raise SystemExit("%s: header does not state %s: %s" % (self.files["kit/trajectory_kf.csv"]["path"],
                                                                    FRAME_CONVENTION, hdr))
        n_badrot = 0
        if "kit/trajectory_frames.csv" in self.files:
            with open(self.files["kit/trajectory_frames.csv"]["path"], encoding="utf-8") as f:
                for ln in f:
                    if ln.startswith("#") or not ln.strip():
                        continue
                    c = ln.strip().split(",")
                    if len(c) < 9:
                        continue
                    T = _T_from_row(c)
                    if not np.isfinite(T).all() or not _rot_ok(T[:3, :3], 1e-4):
                        n_badrot += 1
                        continue
                    fi = int(c[8])
                    self.t[fi], self.T[fi], self.state[fi] = float(c[0]), T, (c[9] if len(c) > 9 else "tracking")
        for fi in kfs:                                    # keyframe poses (final) win over the per-frame file
            self.t[fi], self.T[fi] = kf_t[fi], kf_T[fi]
            self.state.setdefault(fi, "tracking")
        self.frames = sorted(self.T)
        self.kf_frames = sorted(set(kfs))
        self.checks = dict(source="kit/trajectory_kf.csv", header=hdr, quaternion="x, y, z, w (W LAST; kit/writer.py _q_from_R)",
                           kf_rule="every keyframe of kit/trajectory_kf.csv", n_bad_rotation=n_badrot,
                           n_duplicate_kf=len(kfs) - len(set(kfs)))

    def sequence_folder(self):
        return self.manifest.get("dataset")

    def gravity(self):
        s = self.summary or {}
        g = ((s.get("global_ba") or {}).get("inertial") or {}).get("g_w")
        if g is not None:
            return g, "-g_w of the final global BA (summary.json global_ba.inertial.g_w)"
        return None, "no global_ba.inertial.g_w in summary.json"

    def engine_identity(self):
        s = self.summary or {}
        return dict(engine=s.get("engine"), layout="xclean run", manifest_keys=sorted(self.manifest)[:20])


def _T_from_row(c):
    from scipy.spatial.transform import Rotation as Rot
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([float(x) for x in c[4:8]]).as_matrix()       # (x, y, z, w): W LAST
    T[:3, 3] = [float(x) for x in c[1:4]]
    return T
