"""`stimp`: the pan matrix profile, one exact GPU matrix profile per window size.

The pan matrix profile (SKIMP, DOI 10.1109/ICBK.2019.00031, Table 2) stacks
the matrix profiles of a range of window sizes. The window sizes are visited
in the breadth-first (level) order of a balanced binary search tree over the
sorted range, so a few ``update()`` calls already sketch the whole range and
every further call refines it.

The semantics re-implement STUMPY's ``stimp``, ``aamp_stimp`` and
``gpu_stimp`` (STUMPY, Copyright 2019 TD Ameritrade, BSD-3-Clause): the window
range and its ordering, the ``PAN`` row layout, and the normalize, contrast,
binarize, clip and repeat steps of ``pan()``. This is our own code written
from that description; STUMPY is not imported at runtime. Every update computes
one exact profile, i.e. ``stumpy.stimp(..., percentage=1.0, pre_scrump=False)``,
which is what ``stumpy.gpu_stimp`` computes; there is no SCRIMP approximation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._preprocess import check_series, excl_zone_denom
from ._stump import _check_device_id, _stump

IsConstantFunc = Callable[[np.ndarray, int], ArrayLike]


def _bfs_order(n: int) -> np.ndarray:
    """Level-order positions of a balanced binary search tree over ``range(n)``.

    The root of a half-open interval ``[lo, hi)`` is its midpoint
    ``(lo + hi) // 2``; its children cover ``[lo, mid)`` and ``[mid + 1, hi)``.
    Reading the nodes level by level, left to right, gives the order in which
    STUMPY's ``core._bfs_indices`` visits the window sizes (e.g. ``n = 10``:
    5, 2, 8, 1, 4, 7, 9, 0, 3, 6).
    """
    order = np.empty(n, dtype=np.int64)
    filled = 0
    level = [(0, n)] if n > 0 else []
    while level:
        below = []
        for lo, hi in level:
            mid = (lo + hi) // 2
            order[filled] = mid
            filled += 1
            if lo < mid:
                below.append((lo, mid))
            if mid + 1 < hi:
                below.append((mid + 1, hi))
        level = below
    return order


def _max_window_size(n: int, denom) -> int:
    """STUMPY's ``core.get_max_window_size``: the largest self-join window
    for which a series of length ``n`` still has non-trivial neighbors under
    the exclusion-zone denominator ``denom`` (floor-division semantics, as
    STUMPY evaluates it, including a non-integer denominator)."""
    return int(n - np.floor((n + (denom - 1)) // (denom + 1))) - 1


def _as_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"`{name}` must be an integer but found {type(value)}.")
    return int(value)


class stimp:
    """Pan matrix profile computed on the GPU; drop-in for ``stumpy.stimp``.

    Every :meth:`update` computes one exact matrix profile with
    :func:`mlx_stump.stump` (a self-join, ``k=1``) for the next window size
    in breadth-first order, so the result equals
    ``stumpy.stimp(T, min_m, max_m, step, percentage=1.0, pre_scrump=False,
    ...)`` and ``stumpy.gpu_stimp``; STUMPY's approximate SCRIMP path
    (``percentage < 1``) is not offered. ``normalize=False`` follows
    ``stumpy.aamp_stimp`` (``percentage=1.0, pre_scraamp=False``) and
    supports ``p=2.0`` only; ``p`` is ignored when ``normalize=True``.

    Window sizes: with ``max_m=None`` they run from ``min_m`` to
    ``max(min_m + 1, max_window)`` in steps of ``step``; otherwise
    ``min_m``/``max_m`` are swapped if needed and the range is clamped to
    ``[3, max_window]``. ``max_window`` is STUMPY's largest self-join window
    for ``len(T)`` under ``stumpy.config.STUMPY_EXCL_ZONE_DENOM`` (read at
    construction). As in STUMPY, a ``max_m=None`` range is not clamped, so an
    out-of-range window size raises only when its update runs.

    ``T_subseq_isconstant_func(a, w)`` flags constant windows for every
    window size (default: a window is constant when its min equals its max).
    It receives the series with inf replaced by NaN and must return a
    boolean array of length ``len(a) - w + 1``. With ``normalize=False`` the
    flags do not affect raw distances, so the function is called only once,
    at the first update, to validate it (STUMPY's ``aamp_stimp`` does not
    accept it at all).

    Attributes ``PAN_`` (the transformed pan matrix profile, see
    :meth:`pan`), ``M_`` (window sizes in breadth-first order) and ``P_``
    (the raw profiles in that order) follow STUMPY. The class keeps a
    ``(len(M_), len(T))`` float64 array, like STUMPY.

    ``P_`` agrees with STUMPY within float tolerance. The binary ``PAN_`` is
    identical to STUMPY's on noisy data (random walks, sine plus noise), but
    its contrast step ranks every value, so on series with large groups of
    exactly tied distances (exactly periodic or discrete-valued data, or
    constant runs with ``normalize=False``) tied cells can be ranked in a
    different order and some binary cells differ; see :meth:`pan`.
    """

    def __init__(
        self,
        T: ArrayLike,
        min_m: int = 3,
        max_m: int | None = None,
        step: int = 1,
        normalize: bool = True,
        p: float = 2.0,
        T_subseq_isconstant_func: IsConstantFunc | None = None,
    ) -> None:
        self._T = check_series(T, "T")  # private float64 copy, like STUMPY's T.copy()
        n = self._T.shape[0]
        min_m = _as_int(min_m, "min_m")
        if max_m is not None:
            max_m = _as_int(max_m, "max_m")
        step = _as_int(step, "step")
        if step < 1:
            raise ValueError(f"`step` must be a positive integer but found {step}.")
        self._normalize = bool(normalize)
        if not self._normalize and p != 2.0:
            raise NotImplementedError(
                "mlx-stump supports p=2.0 only when normalize=False; "
                f"found p={p}. Use stumpy.aamp_stimp for other p-norms."
            )
        if T_subseq_isconstant_func is not None and not callable(T_subseq_isconstant_func):
            raise ValueError(
                "`T_subseq_isconstant_func` was expected to be a callable function "
                f"but {type(T_subseq_isconstant_func)} was found."
            )
        # None is STUMPY's default rule (a window is constant iff min == max),
        # which stump applies itself without calling back into Python
        self._T_subseq_isconstant_func = T_subseq_isconstant_func
        # raw distances ignore the flags: a user function is only validated,
        # once, by the first successful update
        self._isconstant_func_checked = False

        max_window = _max_window_size(n, excl_zone_denom())
        if max_m is None:
            M = np.arange(min_m, max(min_m + 1, max_window) + 1, step, dtype=np.int64)
        else:
            lo, hi = sorted((min_m, max_m))
            M = np.arange(max(3, lo), min(max_window, hi) + 1, step, dtype=np.int64)
        if M.shape[0] == 0:
            raise ValueError(
                f"No window size between min_m={min_m} and max_m={max_m} is valid for a "
                f"self-join of a length-{n} series (window sizes must lie in "
                f"[3, {max_window}])."
            )
        self._bfs_indices = _bfs_order(M.shape[0])
        # _M[i] is the window size of the i-th update; its profile is PAN row
        # _bfs_indices[i], i.e. the rows are in ascending window-size order
        self._M = M[self._bfs_indices]
        self._n_processed = 0
        # each row holds one profile of length n - m + 1, then inf padding
        self._PAN = np.full((M.shape[0], n), np.inf, dtype=np.float64)

        if not self._normalize:
            # aamp_stimp scales raw distances by the finite value range of T
            finite = self._T[np.isfinite(self._T)]
            if finite.size == 0:
                raise ValueError("`T` must contain at least one finite value when normalize=False.")
            self._T_min = float(finite.min())
            self._T_max = float(finite.max())

    def update(self) -> None:
        """Compute the matrix profile of the next window size in ``M_``.

        Does nothing once every window size has been processed. An error
        (e.g. an out-of-range window size) leaves the state unchanged.
        """
        if self._n_processed >= self._M.shape[0]:
            return
        m = int(self._M[self._n_processed])
        func = self._T_subseq_isconstant_func
        if not self._normalize and self._isconstant_func_checked:
            func = None  # already validated; raw distances do not use the flags
        # _stump directly (not stump): stacklevel=3 attributes its warnings,
        # and one frame deeper those of its helpers, to the caller of update()
        mp = _stump(
            self._T,
            m,
            T_B=None,
            ignore_trivial=True,
            normalize=self._normalize,
            p=2.0,
            k=1,
            T_A_subseq_isconstant=func,
            T_B_subseq_isconstant=None,
            chunk_size=None,
            stacklevel=3,
        )
        P = mp.P_
        self._PAN[self._bfs_indices[self._n_processed], : P.shape[0]] = P
        self._n_processed += 1
        self._isconstant_func_checked = True

    def _pan_scale(self, done: np.ndarray) -> np.ndarray:
        """Per-row factor that maps a profile to roughly ``[0, 1]``.

        Normalized: the largest z-normalized distance is ``2 sqrt(m)``.
        Raw (``aamp_stimp``): ``(max(T) - min(T)) * m**(1/p)`` over the
        finite values of ``T``, with ``p = 2``.
        """
        ms = self._M[: done.shape[0]]
        if self._normalize:
            return 1.0 / (2.0 * np.sqrt(ms))
        # a constant series divides by zero here, as in STUMPY: its rows
        # become NaN (0 * inf) or 1 after pan()'s cap at 1.0
        with np.errstate(divide="ignore"):
            return 1.0 / (np.abs(self._T_max - self._T_min) * np.power(ms, 1.0 / 2.0))

    def pan(
        self,
        threshold: float = 0.2,
        normalize: bool = True,
        contrast: bool = True,
        binary: bool = True,
        clip: bool = True,
    ) -> np.ndarray:
        """Return the transformed pan matrix profile, ``(len(M_), len(T))``.

        Rows are in ascending window-size order. Missing values (the padding
        past each profile, NaN/inf windows) start as NaN and only the rows
        computed so far are transformed, in this order:

        - ``normalize``: divide each profile by ``2 sqrt(m)``, the largest
          z-normalized distance (``normalize=False`` objects: by
          ``(max(T) - min(T)) sqrt(m)`` over the finite values of ``T``, as
          ``aamp_stimp`` does), and cap it at 1.
        - ``contrast``: replace each value by the logistic function
          ``1 / (1 + exp(-10 (q - threshold)))`` of its quantile ``q`` among
          all computed values (stable ranking, missing values last). Because
          this ranks every value, series with exact-tie groups (exactly
          periodic or discrete-valued data, or constant runs with
          ``normalize=False``) can have tied cells ordered differently from
          STUMPY, whose tied distances differ from ours in the last bits
          (e.g. STUMPY's ``aamp`` gives about 2e-7 for identical windows,
          where we give exactly 0). The contrasted and binary values of those
          cells can then differ, while the profiles (``P_``) still agree
          within float tolerance.
        - ``binary``: 0 where the value is ``<= threshold``, else 1
          (missing values become 1).
        - ``clip``: clip to ``[0, 1]``.

        Every row up to the last computed one then repeats the first
        computed row at or after it (the next larger computed window size),
        which gives the blocky picture of a partially updated profile, and
        every value still missing becomes the array's maximum (all NaN
        before the first update).
        """
        PAN = self._PAN.copy()
        PAN[PAN == np.inf] = np.nan
        done = self._bfs_indices[: self._n_processed]
        if done.shape[0] > 0:
            rows = PAN[done]
            if normalize:
                with np.errstate(invalid="ignore"):  # raw mode, constant T: 0 * inf
                    rows = np.minimum(1.0, rows * self._pan_scale(done)[:, None])
            if contrast:
                # stable ranks over the flattened rows, in update order (the
                # tie order STUMPY uses); NaN sorts after every number
                order = np.argsort(rows, axis=None, kind="stable")
                quantile = np.empty(rows.size, dtype=np.float64)
                quantile[order] = np.linspace(0, 1, rows.size)
                quantile = quantile.reshape(rows.shape)
                rows = 1.0 / (1.0 + np.exp(-10 * (quantile - threshold)))
            if binary:
                rows = np.where(rows <= threshold, 0.0, 1.0)
            if clip:
                rows = np.clip(rows, 0.0, 1.0)
            PAN[done] = rows

            # rows 0..max(done) repeat the first computed row at or after them
            computed = np.sort(done)
            nrepeat = np.diff(computed, prepend=-1)
            PAN[: computed[-1] + 1] = np.repeat(PAN[computed], nrepeat, axis=0)

        missing = np.isnan(PAN)
        if missing.any():
            # fmax ignores NaN; an all-NaN array stays NaN (no RuntimeWarning)
            PAN[missing] = np.fmax.reduce(PAN, axis=None)
        return PAN

    @property
    def PAN_(self) -> np.ndarray:
        """``pan()`` with its defaults: normalized, contrasted, binarized,
        clipped and repeated."""
        return self.pan()

    @property
    def M_(self) -> np.ndarray:
        """All window sizes, in breadth-first (update) order (int64 copy)."""
        return self._M.copy()

    @property
    def P_(self) -> list[np.ndarray]:
        """Raw profiles in breadth-first order, one per ``M_`` entry.

        Each is a float64 array of length ``len(T) - m + 1``; profiles not
        computed yet are all ``inf``. Like STUMPY's, they are views into the
        internal pan array (no extra memory, and a later :meth:`update` fills
        them in place), but read-only, so they cannot corrupt it; use
        ``.copy()`` to edit one.
        """
        n = self._T.shape[0]
        profiles = []
        for row, m in zip(self._bfs_indices, self._M, strict=True):
            view = self._PAN[row, : max(n - int(m) + 1, 0)]
            view.flags.writeable = False
            profiles.append(view)
        return profiles


class gpu_stimp(stimp):
    """:class:`stimp` under ``stumpy.gpu_stimp``'s signature.

    ``device_id`` (an int or a list of ints, as in STUMPY) is validated and
    then ignored: an Apple silicon Mac has one GPU, which MLX already uses.
    """

    def __init__(
        self,
        T: ArrayLike,
        min_m: int = 3,
        max_m: int | None = None,
        step: int = 1,
        device_id: int | Sequence[int] = 0,
        normalize: bool = True,
        p: float = 2.0,
        T_subseq_isconstant_func: IsConstantFunc | None = None,
    ) -> None:
        _check_device_id(device_id)
        super().__init__(
            T,
            min_m=min_m,
            max_m=max_m,
            step=step,
            normalize=normalize,
            p=p,
            T_subseq_isconstant_func=T_subseq_isconstant_func,
        )
