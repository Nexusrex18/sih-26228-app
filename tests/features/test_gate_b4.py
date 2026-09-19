"""Gate B4 — exactly what the gate sentence claims, and nothing more.

> embeddings extract offline under the egress guard; the cache returns a hit on a second
> run and a MISS when `extractor_version` changes; the measured CPU img/s is recorded in
> the plan and the budget tiers are derived from it.

Four claims, four tests, plus the two wiring tests for the B4 decision that embeddings
reach `scan()` here rather than at B5. The measured img/s is deliberately NOT asserted:
throughput is a property of the machine, and a slow runner is not a broken build
(`cva/features/throughput.py` is the instrument; the number lives in the plan).
"""
from __future__ import annotations

import io
import tempfile

import numpy as np
import pytest

from cva.features import vendor
from cva.features.cache import FeatureCache
from cva.features.embed import ArrayEmbeddingIndex, build_extractor, embed_samples

pytestmark = pytest.mark.skipif(
    not (vendor.available("resnet18") or vendor.available("dinov2_vits14")),
    reason="no vendored backbone on disk — Mode A runs `make vendor`")


def png(seed: int, px: int = 64) -> bytes:
    from PIL import Image
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (px, px, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def extractor():
    import torch
    torch.hub.set_dir(tempfile.mkdtemp())
    return build_extractor()


def test_the_cache_hits_on_a_second_run_and_misses_when_the_version_moves(extractor, tmp_path):
    """V8. The version in the key is a CORRECTNESS property: a cache keyed on content
    alone hands back last week's vectors after the extraction procedure changes, and the
    report then attributes findings to a backbone that did not produce them."""
    cache = FeatureCache(root=tmp_path / "features")
    items = [(f"s{i}", png(i)) for i in range(3)]

    embed_samples(items, extractor, cache)
    assert (cache.hits, cache.misses) == (0, 3)

    embed_samples(items, extractor, cache)
    assert cache.hits == 3, "second run over identical bytes must be a hit"

    before = cache.misses
    extractor.extractor_version = extractor.extractor_version + "+bumped"
    try:
        embed_samples(items, extractor, cache)
    finally:
        extractor.extractor_version = extractor.extractor_version.removesuffix("+bumped")
    assert cache.misses == before + 3, "a changed extractor_version must MISS, not hit"


def test_a_corrupt_entry_is_a_miss_and_is_never_read_back(tmp_path):
    cache = FeatureCache(root=tmp_path / "features")
    cache.put("abc", "ext", "1", {"pooled": np.zeros(4, dtype=np.float32)})
    entry = next((tmp_path / "features").rglob("*.npz"))
    entry.write_bytes(b"not an npz")
    assert cache.get("abc", "ext", "1") is None
    assert not entry.exists(), "a corrupt entry is deleted, never silently reused"


def test_both_backbones_load_with_the_network_torn_out(monkeypatch):
    """Proven, not hoped for: every egress path a backbone load could take is replaced
    with a raise before the load runs. `pretrained=False` / `weights=None` is the claim;
    this is the test that the claim is true of the code as written."""
    import urllib.request

    import torch

    torch.hub.set_dir(tempfile.mkdtemp())

    def explode(*a, **k):
        raise AssertionError("a backbone load attempted a network call")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", explode)
    monkeypatch.setattr(torch.hub, "download_url_to_file", explode, raising=False)

    for name in ("dinov2_vits14", "resnet18"):
        if not vendor.available(name):
            continue
        ex = build_extractor(name)
        assert ex.extractor_id.startswith(f"{name}@sha256:"), \
            "extractor_id pins the checkpoint's byte string, never its name (ADR-009)"
        assert ex.dim > 0


def test_knn_is_deterministic_regardless_of_insertion_order():
    """Two identical images are the near-duplicate case this index exists to find. Ranked
    by array order they would follow whichever the loader happened to walk first, and
    V10's empty diff fails on file order."""
    rng = np.random.default_rng(0)
    vecs = {f"s{i}": rng.normal(size=8).astype(np.float32) for i in range(6)}
    vecs["s5"] = vecs["s1"].copy()

    def build(order):
        idx = ArrayEmbeddingIndex("e", "1", 8)
        for sid in order:
            idx.add(sid, vecs[sid], None)
        return idx

    forward = build(sorted(vecs))
    reverse = build(sorted(vecs, reverse=True))
    assert forward.knn("s1", 3) == reverse.knn("s1", 3)
    assert forward.knn("s1", 1)[0][0] == "s5"


# --- the B4 wiring decision: embeddings reach scan(), lazily ---------------

class _Spy:
    id = "data.spy"
    version = "1.0.0"
    requires: set = set()
    optional: set = set()
    attack_classes = {"near_duplicate_flooding"}
    seen: list = []

    def detect(self, dataset, embeddings, model, ctx=None):
        _Spy.seen.append(embeddings)
        return []


class _Dataset:
    samples: list = []
    categories: list = []

    def annotations(self):
        return []

    def capabilities(self):
        from cva.core.capability import CapabilitySet
        return CapabilitySet()


def _run(monkeypatch, dataset):
    from cva.core.orchestrator import RunContext, scan
    _Spy.seen = []
    ctx = RunContext(dataset=dataset)
    scan(None, ctx, "deep", registries=({}, {"data.spy": _Spy}))
    return _Spy.seen


def test_no_backbone_is_loaded_when_no_detector_row_needs_one(monkeypatch):
    """The P0 gate runs with zero detectors registered. Loading an 88 MB backbone to feed
    nothing would put a torch import and a minute on the one path with no use for it."""
    import cva.core.orchestrator as orch
    called = []
    monkeypatch.setattr(orch, "build_embeddings",
                        lambda ctx, prof: (called.append(1), (None, None))[1])
    from cva.core.orchestrator import RunContext, scan

    class _Check:
        id = "stub.ok"
        version = "1.0.0"
        requires: set = set()
        optional: set = set()
        attack_classes = {"near_duplicate_flooding"}

        def check(self, model, ctx):
            return []

    # A runnable row exists, so the loop really iterates — without that this would pass
    # against an empty plan and assert nothing.
    scan(None, RunContext(dataset=_Dataset()), "deep",
         registries=({"stub.ok": _Check}, {}))
    assert not called, "no detector row ran, so no backbone should have been built"


def test_a_detector_receives_the_index_and_the_reason_when_there_is_none(monkeypatch):
    import cva.core.orchestrator as orch

    monkeypatch.setattr(orch, "build_embeddings",
                        lambda ctx, prof: (None, "backbone did not load in this test"))
    seen = _run(monkeypatch, _Dataset())
    assert seen == [None], "the detector must still run; `embeddings is None` is its path"

    sentinel = ArrayEmbeddingIndex("dinov2_vits14@sha256:deadbeef", "1.0.0", 384)
    monkeypatch.setattr(orch, "build_embeddings", lambda ctx, prof: (sentinel, None))
    assert _run(monkeypatch, _Dataset()) == [sentinel]


def test_the_manifest_records_whether_the_distributor_published_the_digest():
    """A trust-on-first-use pin proves the bytes have not changed SINCE, not that they
    were the intended bytes to begin with. The manifest must not imply more."""
    rows = {r["artefact"]: r for r in vendor.manifest()}
    assert rows["dinov2_vits14"]["digest_published_by_distributor"] is False
    assert rows["resnet18"]["digest_published_by_distributor"] is True
