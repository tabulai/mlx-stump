"""The public peak estimate includes optional dense-repair storage."""

from mlx_stump import _engine


def test_estimate_charges_bounded_normalized_repair_cache(monkeypatch):
    # Make assembly small enough to expose the repair term directly. The
    # production refinement budget is larger, so this path is usually hidden
    # beneath the assembly or automatic sweep phase in the maximum.
    monkeypatch.setattr(_engine, "_REFINE_MEM_BUDGET", 1 << 20)
    l, m = 20_000, 256
    cache = l * (m + 1) * 8
    assert cache <= _engine._REPAIR_NORMALIZED_CACHE_MAX_BYTES
    with_cache = _engine.estimated_peak_bytes(l, m, self_join=True, chunk_size=1)
    without_cache = _engine.estimated_peak_bytes(l, m, self_join=False, chunk_size=1)
    assert with_cache >= (
        _engine.resident_block_bytes(l, m) + l * (24 + 64) + (32 << 20) + cache
    )
    assert with_cache > without_cache
