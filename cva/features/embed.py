"""Embedding extraction on the vendored backbone, and the `EmbeddingIndex` the Module A
detectors consume.

Two implementations, per ADR-008's ">=2 implementations" rule: DINOv2 ViT-S/14 as the
primary and torchvision's ResNet-18 as the fallback. Both load with **no network access
whatsoever** — `pretrained=False` / `weights=None` plus a local `load_state_dict` from a
digest-pinned file. The hub and torchvision defaults (`pretrained=True`,
`weights=IMAGENET1K_V1`) both call `load_state_dict_from_url`, which egresses; B6's guard
kills them, which is the correct outcome but a late one. They are never called here.

Why patch tokens are not optional garnish: ADR-009 chose DINOv2 over a pooled-only
backbone precisely for dense per-patch features, because Module A's patch-saliency scan
needs per-region representations. The ResNet-18 fallback therefore also exposes a patch
grid — its last conv feature map, 7x7 at 224 px — rather than pooled vectors only. A
fallback that silently dropped them would degrade the evidence artefact without saying so.
"""
from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cva.features import vendor
from cva.features.cache import FeatureCache, content_sha256

#: Bumped whenever the extraction PROCEDURE changes — preprocessing, pooling, dtype,
#: anything that moves a vector without moving the checkpoint. It is half the cache key
#: for that reason (§B4, V8).
EXTRACTOR_VERSION = "1.0.0"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_PX = 224          # 224 = 14 x 16: a whole number of ViT-S/14 patches


class BackboneUnavailable(RuntimeError):
    """Neither backbone could be loaded. Raised, never swallowed: a scan that silently
    ran without embeddings reports every embedding-derived attack class as assessed."""


def preprocess(img_bytes: bytes) -> np.ndarray:
    """Bytes -> CHW float32, resized to INPUT_PX and ImageNet-normalised.

    Deterministic by construction: a fixed square resize with a fixed resampling filter,
    no random crop, no aspect-ratio jitter. V10's empty diff depends on this.
    """
    from PIL import Image
    with Image.open(io.BytesIO(img_bytes)) as src:
        im = src.convert("RGB").resize((INPUT_PX, INPUT_PX), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


@dataclass
class _TorchExtractor:
    """Shared batching and cache plumbing. Subclasses supply `_forward`."""

    extractor_id: str = ""
    extractor_version: str = EXTRACTOR_VERSION
    dim: int = 0
    model: Any = None
    batch_size: int = 16

    def _forward(self, batch: Any) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def embed_batch(self, images: Sequence[bytes]) -> list[dict[str, np.ndarray]]:
        import torch
        x = np.stack([preprocess(b) for b in images])
        with torch.no_grad():
            pooled, patches = self._forward(torch.from_numpy(x))
        return [{"pooled": pooled[i], "patches": patches[i]} for i in range(len(images))]


class Dinov2Extractor(_TorchExtractor):
    """DINOv2 ViT-S/14. 384-dim CLS token, 256 patch tokens at 224 px."""

    def __init__(self) -> None:
        import torch
        art = vendor.ARTEFACTS["dinov2_vits14"]
        weights = vendor.local_path("dinov2_vits14")       # verifies the digest or raises
        if not vendor.repo_available():
            raise BackboneUnavailable(
                f"dinov2 source snapshot absent at {vendor.DINOV2_REPO_DIR}; "
                "`source='local'` localises the CODE only, so the weights alone are not "
                "enough (backend_plan §5.4)")
        model = torch.hub.load(str(vendor.DINOV2_REPO_DIR), "dinov2_vits14",
                               source="local", pretrained=False)
        model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"),
                              strict=True)
        model.eval()
        # The id pins the checkpoint's BYTE STRING, not its name (ADR-009): the same
        # repository distributes XRay-DINO and Cell-DINO under a different licence.
        super().__init__(extractor_id=f"dinov2_vits14@sha256:{art.sha256}",
                         dim=384, model=model)

    def _forward(self, batch: Any) -> tuple[np.ndarray, np.ndarray]:
        out = self.model.forward_features(batch)
        return (out["x_norm_clstoken"].numpy().astype(np.float32),
                out["x_norm_patchtokens"].numpy().astype(np.float32))


class ResNet18Extractor(_TorchExtractor):
    """The ADR-008 second implementation. 512-dim pooled, 7x7=49 patch tokens at 224 px."""

    def __init__(self) -> None:
        import torch
        from torchvision.models import resnet18
        art = vendor.ARTEFACTS["resnet18"]
        weights = vendor.local_path("resnet18")
        # weights=None, NOT weights=IMAGENET1K_V1: the enum form downloads.
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"),
                              strict=True)
        model.eval()
        self.trunk = torch.nn.Sequential(*list(model.children())[:-2])   # -> (B,512,7,7)
        super().__init__(extractor_id=f"resnet18@sha256:{art.sha256}", dim=512, model=model)

    def _forward(self, batch: Any) -> tuple[np.ndarray, np.ndarray]:
        fmap = self.trunk(batch)                       # (B, 512, 7, 7)
        pooled = fmap.mean(dim=(2, 3))                 # (B, 512)
        b, c, h, w = fmap.shape
        patches = fmap.reshape(b, c, h * w).transpose(1, 2)    # (B, 49, 512)
        return pooled.numpy().astype(np.float32), patches.numpy().astype(np.float32)


def build_extractor(prefer: str | None = None) -> _TorchExtractor:
    """DINOv2 first, ResNet-18 second, and an honest error third.

    The fallback is a real fallback and not a silent substitution: the chosen
    `extractor_id` travels into every cache key, every EmbeddingIndex and every finding
    that cites one, so a report produced on the fallback says so on its face.
    """
    order = [prefer] if prefer else ["dinov2_vits14", "resnet18"]
    builders: dict[str, Any] = {"dinov2_vits14": Dinov2Extractor,
                                "resnet18": ResNet18Extractor}
    reasons: list[str] = []
    for name in order:
        try:
            built: _TorchExtractor = builders[name]()
        except Exception as exc:
            reasons.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            return built
    raise BackboneUnavailable(
        "no embedding backbone could be loaded; embedding-derived attack classes are "
        "UNAVAILABLE for this scan, not assessed.\n  " + "\n  ".join(reasons))


@dataclass
class ArrayEmbeddingIndex:
    """An in-memory `EmbeddingIndex` (types.py §7.7b) over already-extracted vectors."""

    extractor_id: str
    extractor_version: str
    dim: int
    _pooled: dict[str, np.ndarray] = field(default_factory=dict)
    _patches: dict[str, np.ndarray] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _unit: np.ndarray | None = field(default=None, repr=False)

    def add(self, sample_id: str, pooled: np.ndarray, patches: np.ndarray | None) -> None:
        if sample_id not in self._pooled:
            self._order.append(sample_id)
        self._pooled[sample_id] = np.asarray(pooled, dtype=np.float32)
        if patches is not None:
            self._patches[sample_id] = np.asarray(patches, dtype=np.float32)
        self._unit = None

    def __len__(self) -> int:
        return len(self._order)

    def vector(self, sample_id: str) -> np.ndarray | None:
        return self._pooled.get(sample_id)

    def vectors(self, sample_ids: Sequence[str]) -> np.ndarray:
        if not sample_ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._pooled[s] for s in sample_ids])

    def patch_tokens(self, sample_id: str) -> np.ndarray | None:
        return self._patches.get(sample_id)

    def _unit_matrix(self) -> np.ndarray:
        if self._unit is None:
            m = self.vectors(self._order)
            n = np.linalg.norm(m, axis=1, keepdims=True)
            # A zero vector has no direction; leaving the norm at 0 would produce NaN and
            # NaN sorts unpredictably, which would make knn() non-deterministic.
            self._unit = m / np.where(n == 0, 1.0, n)
        return self._unit

    def knn(self, sample_id: str, k: int) -> list[tuple[str, float]]:
        if sample_id not in self._pooled or len(self._order) < 2:
            return []
        u = self._unit_matrix()
        q = self._pooled[sample_id]
        qn = float(np.linalg.norm(q))
        sims = u @ (q / (qn if qn else 1.0))
        i_self = self._order.index(sample_id)
        # Tie-break on sample_id, not on array order: two identical images (the
        # near-duplicate case this index exists to find) otherwise rank by whichever the
        # loader happened to walk first, and V10's empty diff fails on file order.
        ranked = sorted(((self._order[i], float(sims[i])) for i in range(len(self._order))
                         if i != i_self), key=lambda t: (-t[1], t[0]))
        return ranked[:k]


def embed_samples(items: Sequence[tuple[str, bytes]], extractor: _TorchExtractor,
                  cache: FeatureCache | None = None,
                  keep_patches: bool = True) -> ArrayEmbeddingIndex:
    """Extract (or re-use) one vector per sample, in a deterministic order."""
    cache = FeatureCache() if cache is None else cache
    idx = ArrayEmbeddingIndex(extractor.extractor_id, extractor.extractor_version,
                              extractor.dim)
    pending: list[tuple[str, str, bytes]] = []

    def flush() -> None:
        for start in range(0, len(pending), extractor.batch_size):
            chunk = pending[start:start + extractor.batch_size]
            for (sid, csum, _), arrays in zip(
                    chunk, extractor.embed_batch([b for _, _, b in chunk]), strict=True):
                cache.put(csum, extractor.extractor_id, extractor.extractor_version, arrays)
                idx.add(sid, arrays["pooled"],
                        arrays.get("patches") if keep_patches else None)
        pending.clear()

    for sample_id, data in items:
        csum = content_sha256(data)
        got = cache.get(csum, extractor.extractor_id, extractor.extractor_version)
        if got is not None:
            idx.add(sample_id, got["pooled"], got.get("patches") if keep_patches else None)
        else:
            pending.append((sample_id, csum, data))
    flush()
    return idx


def embed_paths(paths: Sequence[Path | str], extractor: _TorchExtractor,
                cache: FeatureCache | None = None) -> ArrayEmbeddingIndex:
    items = [(str(p), Path(p).read_bytes()) for p in sorted(map(str, paths))]
    return embed_samples(items, extractor, cache)


def embed_dataset(dataset: Any, extractor: _TorchExtractor,
                  cache: FeatureCache | None = None,
                  keep_patches: bool = True) -> ArrayEmbeddingIndex:
    """Index a loaded `Dataset`, reusing `Sample.content_sha256` as the cache key.

    The loader already hashed every file; re-hashing here would read the whole corpus a
    second time per scan for a digest we are holding.
    """
    cache = FeatureCache() if cache is None else cache
    idx = ArrayEmbeddingIndex(extractor.extractor_id, extractor.extractor_version,
                              extractor.dim)
    pending: list[tuple[str, str, bytes]] = []

    def flush() -> None:
        for start in range(0, len(pending), extractor.batch_size):
            chunk = pending[start:start + extractor.batch_size]
            for (sid, csum, _), arrays in zip(
                    chunk, extractor.embed_batch([b for _, _, b in chunk]), strict=True):
                cache.put(csum, extractor.extractor_id, extractor.extractor_version, arrays)
                idx.add(sid, arrays["pooled"],
                        arrays.get("patches") if keep_patches else None)
        pending.clear()

    for s in sorted(dataset.samples, key=lambda s: s.sample_id):
        csum = s.content_sha256 or content_sha256(Path(s.path).read_bytes())
        got = cache.get(csum, extractor.extractor_id, extractor.extractor_version)
        if got is not None:
            idx.add(s.sample_id, got["pooled"],
                    got.get("patches") if keep_patches else None)
        else:
            pending.append((s.sample_id, csum, Path(s.path).read_bytes()))
    flush()
    return idx
