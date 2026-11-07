# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Epoch-shuffled view sampler (BUILDER 5.4: every photo once per pass, in a fresh order each pass).

The permutation of epoch e is a pure function of (seed, e), so the whole sampler state is (epoch, pos) and a resume
reproduces the exact sequence without storing permutations.
"""
from __future__ import annotations

import numpy as np


class EpochSampler:
    def __init__(self, n, seed=0, epoch=0, pos=0):
        if n <= 0:
            raise ValueError("no training views")
        self.n = int(n)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.pos = int(pos)
        self._perm = self._make(self.epoch)

    def _make(self, e):
        return np.random.default_rng([self.seed, 1009, int(e)]).permutation(self.n)

    def next(self):
        if self.pos >= self.n:
            self.epoch += 1
            self.pos = 0
            self._perm = self._make(self.epoch)
        i = int(self._perm[self.pos])
        self.pos += 1
        return i

    def state_dict(self):
        return dict(n=self.n, seed=self.seed, epoch=self.epoch, pos=self.pos)

    @classmethod
    def from_state(cls, d):
        return cls(d["n"], d["seed"], d["epoch"], d["pos"])
