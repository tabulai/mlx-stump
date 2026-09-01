"""STUMPY-compatible matrix profile array: object ndarray plus P_/I_ attributes."""

from __future__ import annotations

import numpy as np


class mparray(np.ndarray):
    """An ndarray with convenience accessors matching ``stumpy.mparray``.

    Column layout (k = number of nearest neighbors): ``[:, :k]`` profile
    values, ``[:, k:2k]`` neighbor indices, ``[:, 2k]`` left indices,
    ``[:, 2k+1]`` right indices.
    """

    def __new__(cls, input_array, m, k, excl_zone_denom):
        obj = np.asarray(input_array).view(cls)
        obj._m = m
        obj._k = k
        obj._excl_zone_denom = excl_zone_denom
        return obj

    def __array_finalize__(self, obj):
        if obj is None:  # pragma: no cover
            return
        self._m = getattr(obj, "_m", None)
        self._k = getattr(obj, "_k", None)
        self._excl_zone_denom = getattr(obj, "_excl_zone_denom", None)

    @property
    def P_(self) -> np.ndarray:
        if self._k == 1:
            return self[:, : self._k].flatten().astype(np.float64)
        return self[:, : self._k].astype(np.float64)

    @property
    def I_(self) -> np.ndarray:
        if self._k == 1:
            return self[:, self._k : 2 * self._k].flatten().astype(np.int64)
        return self[:, self._k : 2 * self._k].astype(np.int64)

    @property
    def left_I_(self) -> np.ndarray:
        if self._k == 1:
            return self[:, 2 * self._k].flatten().astype(np.int64)
        return self[:, 2 * self._k].astype(np.int64)

    @property
    def right_I_(self) -> np.ndarray:
        if self._k == 1:
            return self[:, 2 * self._k + 1].flatten().astype(np.int64)
        return self[:, 2 * self._k + 1].astype(np.int64)
