# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Shared helpers: atomic writes, hashing, logging. No torch import here.

Every file the builder writes goes through atomic_* (tmp + fsync + os.replace), so a power cut never leaves a half
file under its final name. Writes are fenced: set_write_root(<--out dir>) is called by the pipeline, and a write
outside that root (or an extra root a test adds) raises PermissionError.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

# --------------------------------------------------------------------------------------------- write fence
_WRITE_ROOTS = []


def set_write_root(*roots):
    """The only directories the builder may write into (the --out dir of the build)."""
    del _WRITE_ROOTS[:]
    _WRITE_ROOTS.extend(os.path.abspath(r) for r in roots if r)


def add_write_root(root):
    if root and os.path.abspath(root) not in _WRITE_ROOTS:
        _WRITE_ROOTS.append(os.path.abspath(root))


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def assert_writable(path):
    """Refuse any write outside the write roots (set_write_root). No root set = refuse everything."""
    a = _norm(path)
    for r in _WRITE_ROOTS:
        s = _norm(r)
        if a == s or a.startswith(s + os.sep):
            return path
    raise PermissionError("xclean-maps: refusing to write outside %s: %s" % (_WRITE_ROOTS, path))


def ensure_dir(d):
    assert_writable(d)
    os.makedirs(d, exist_ok=True)
    return d


def _replace_retry(tmp, path, tries=20, wait=0.05):
    """os.replace with retries: on Windows a reader holding the target open (the live viewer) makes it fail."""
    last = None
    for _ in range(tries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError as e:   # pragma: no cover - timing dependent
            last = e
            time.sleep(wait)
    raise last


def atomic_write_bytes(path, data, retries=20):
    assert_writable(path)
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _replace_retry(tmp, path, tries=retries)


def atomic_write_text(path, text, retries=20):
    atomic_write_bytes(path, text.encode("utf-8"), retries=retries)


def np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError("not JSON serialisable: %s" % type(o))


def atomic_json(path, obj, indent=1, retries=20):
    atomic_write_text(path, json.dumps(obj, indent=indent, default=np_default), retries=retries)


def read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_npy(path, arr):
    assert_writable(path)
    tmp = "%s.tmp%d.npy" % (path, os.getpid())
    np.save(tmp, arr)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    _replace_retry(tmp, path)


def remove_if_exists(path):
    """Stale-artefact rule: delete an output BEFORE regenerating it."""
    if os.path.isdir(path):
        import shutil
        assert_writable(path)
        shutil.rmtree(path)
    elif os.path.exists(path):
        assert_writable(path)
        os.remove(path)


# --------------------------------------------------------------------------------------------- hashing
def sha1_file(path, bufsize=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha1_bytes(b):
    return hashlib.sha1(b).hexdigest()


def sha1_json(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=np_default).encode("utf-8")).hexdigest()


def file_info(path):
    return {"path": path.replace("\\", "/"), "bytes": os.path.getsize(path), "sha1": sha1_file(path)}


# --------------------------------------------------------------------------------------------- logging
class Log:
    def __init__(self, path=None, prefix=""):
        self.path = path
        self.prefix = prefix
        self.f = None
        if path:
            ensure_dir(os.path.dirname(path))
            self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = "[%s] %s%s" % (time.strftime("%Y-%m-%d %H:%M:%S"), self.prefix, msg)
        try:
            print(line, flush=True)
        except (OSError, ValueError):   # a closed console must never stop the build
            pass
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


# --------------------------------------------------------------------------------------------- process hygiene
def low_priority(threads=None):
    """Cap BLAS/OMP threads of THIS process (xclean-maps does not change process priority; the internal builder did)."""
    if threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[k] = str(threads)


def hide_gpu_for_cpu_work():
    """Phase-1 rule: every CPU python hides the GPU (an EMPTY value does not hide it; -1 does)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True


def record_pid(path, role, extra=None):
    """Append this process to a pids.json in the build's work dir (for the user's own bookkeeping)."""
    d = read_json(path, default={"pids": []})
    d["pids"].append(dict(pid=os.getpid(), role=role, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          cmdline=" ".join(sys.argv), **(extra or {})))
    atomic_json(path, d)


def stats(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"p1": float(np.percentile(v, 1)), "p5": float(np.percentile(v, 5)), "p10": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "n": int(v.size)}


def rnd(x, k=4):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    r = round(x, k)
    return 0.0 if r == 0 else r
