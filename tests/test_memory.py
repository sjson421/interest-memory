import zlib
from datetime import datetime, timedelta, timezone

import numpy as np

from interest_memory import InterestMemory

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def bag_of_words(texts):
    """Offline stand-in for fastembed: texts sharing words are similar."""
    vecs = np.zeros((len(texts), 64))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            vecs[i, zlib.crc32(word.encode()) % 64] += 1
    return vecs


def memory(**kw):
    return InterestMemory(":memory:", embed=bag_of_words, **kw)


def test_empty_memory_scores_zero():
    assert memory().score(["agent tool use"], T0) == [0.0]


def test_seeds_rank_similar_text_higher():
    mem = memory()
    mem.add_seeds(["agent tool use"])
    near, far = mem.score(["agent tool use benchmark", "protein folding"], T0)
    assert near > far


def test_rejection_lowers_similar_text_and_forget_restores_it():
    mem = memory()
    mem.add_seeds(["agent tool use"])
    before = mem.score(["agent tool use benchmark"], T0)[0]
    mem.record("2601.00001", "agent tool use benchmark", -1.0, T0)
    assert mem.score(["agent tool use benchmark"], T0)[0] < before
    mem.forget("2601.00001")
    assert mem.score(["agent tool use benchmark"], T0)[0] == before


def test_weight_halves_at_half_life():
    mem = memory(half_life_days=30)
    mem.record("a", "retrieval augmented generation", 1.0, T0)
    fresh = mem.score(["retrieval augmented generation"], T0)[0]
    aged = mem.score(["retrieval augmented generation"], T0 + timedelta(days=30))[0]
    assert np.isclose(aged, fresh / 2)


def test_record_replaces_earlier_label():
    mem = memory()
    mem.record("a", "agent evals", 0.15, T0)
    mem.record("a", "agent evals", -1.0, T0)
    assert mem.score(["agent evals"], T0)[0] < 0


def test_profile_returns_latest_version():
    mem = memory()
    assert mem.profile() == ""
    mem.set_profile("v1")
    mem.set_profile("v2")
    assert mem.profile() == "v2"


def test_opposite_rejection_does_not_raise_score():
    mem = InterestMemory(":memory:", embed=lambda texts: np.array([[1.0, 0.0] if t == "a" else [-1.0, 0.0] for t in texts]))
    mem.add_seeds([])
    mem.record("x", "a", -1.0, T0)
    assert mem.score(["b"], T0) == [0.0]
