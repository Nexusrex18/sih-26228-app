"""The concrete `Dataset` both format loaders produce, and the contributor precedence.

`Dataset` is a Protocol in `core/types.py`; this is the one implementation the loaders
return. Keeping it here rather than in `core/` is the plug-in boundary doing its job —
`core/` declares the shape every seat builds against, `loaders/` owns how it gets filled.

PS §2.2.1 names three grouping levels, and this module is where a loader learns all three.

Contributor (`Sample.contributor`, with its tier in `contributor_source`) is resolved by
`resolve_contributor`, in this order: 1 `contributors.yaml` sidecar, 2 a `contrib_*` /
`contributor_*` directory, 3 the format's own field (COCO `images[].source`), 5 nothing. Tier 4
(EXIF clustering) is a dataset-wide operation and lives with its detector.

Batch (`Sample.batch`) and source (`Sample.source_meta["source"]`) are resolved by
`resolve_grouping`, in the same spirit as tier 2: a directory is evidence only if it SAYS it is.
- a path component starting `batch_` or `batch-` (case-insensitive, non-empty remainder) sets the
  batch to the remainder; one starting `source_` or `source-` sets the source;
- only components BELOW the dataset root and above the file name are read, so the root's own
  name (`/data/batch_root`) and the file name never count;
- when several components match, the one NEAREST the file wins (`batch_a/batch_b/x.png` is batch
  `b`): the innermost folder is the most specific statement about where the file came from;
- COCO `images[].batch` sets the batch and WINS over a `batch_*` directory, being an explicit
  field on the record rather than an inference from layout;
- COCO `images[].source` is NOT read into `source_meta["source"]`. It is already the contributor's
  tier 3 declaration, and mapping it twice would emit identical contributor and source rows.
- the three mechanisms are independent: one file may sit under `contrib_x/batch_a/source_s/`.
A value derived this way is supplier-controlled text that becomes a report `group_value`, so it
is kept as a plain string, control and format characters are replaced with `_`, it is trimmed,
and it is cut to `GROUPING_MAX_LEN` characters. A remainder that is empty after that is ignored.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cva.core.capability import Capability, CapabilitySet
from cva.core.types import Annotation, Category, ContributorSource, Sample
from cva.loaders.safety import check_pixel_budget, safe_join

#: Tier 1 of the precedence. Deliberately a fixed name: a configurable one would let a
#: dataset nominate its own contributor file, which is the supplier attesting to their own
#: provenance — the claim the sidecar exists to make independently auditable.
SIDECAR_NAME = "contributors.yaml"

#: Tier 2. A directory is contributor evidence only if it SAYS it is; any other layout
#: (train/val, class folders) says nothing about who sent what, and reading a contributor
#: out of `images/` would invent one for every dataset on earth.
CONTRIB_DIR_PREFIXES = ("contrib_", "contrib-", "contributor_", "contributor-")

#: The batch and source levels, by the same rule as tier 2: the directory must say so itself.
BATCH_DIR_PREFIXES = ("batch_", "batch-")
SOURCE_DIR_PREFIXES = ("source_", "source-")

#: A batch or source value is a `group_value` in the report, taken from a name the supplier
#: chose; the cap keeps one hostile directory name from becoming a page-wide row label.
GROUPING_MAX_LEN = 128


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class InMemoryDataset:
    samples: list[Sample] = field(default_factory=list)
    categories: list[Category] = field(default_factory=list)
    root: Path | None = None
    #: Why a capability is absent, recorded at probe time — `UNAVAILABLE` is not a skip and
    #: the report has to name what was missing.
    notes: list[tuple[Capability, str]] = field(default_factory=list)

    def __len__(self) -> int:
        # Module A's detectors take `len(dataset)`; the frozen `Dataset` protocol declares no
        # `__len__`, so the protocol stays put and the concrete loader output supplies it.
        return len(self.samples)

    def sample(self, sample_id: str) -> Sample:
        """Lookup by id, which Module A's cluster detectors need. Indexed on first use and
        rebuilt if the sample list changed length, so a loader that appends stays correct."""
        idx: dict[str, Sample] | None = self.__dict__.get("_by_id")
        if idx is None or len(idx) != len(self.samples):
            idx = {s.sample_id: s for s in self.samples}
            self.__dict__["_by_id"] = idx
        return idx[sample_id]

    def category_name(self, category_id: int) -> str:
        for c in self.categories:
            if c.category_id == category_id:
                return c.name
        return str(category_id)

    def annotations(self) -> list[Annotation]:
        """DERIVED on every call, never stored.

        Two stores can disagree; a store and a projection cannot. This is the projection,
        and it sits OUTSIDE the freeze so its shape can change without whole-team
        agreement — which is the entire reason the line is drawn here.
        """
        out: list[Annotation] = []
        for s in self.samples:
            for i, lb in enumerate(s.labels):
                out.append(Annotation(
                    annotation_id=lb.label_id or f"{s.sample_id}:{i}",
                    sample_id=s.sample_id,
                    category_id=lb.category_id,
                    bbox=lb.bbox,
                    iscrowd=lb.iscrowd,
                ))
        return out

    def capabilities(self) -> CapabilitySet:
        """PROBED, not asserted — and the probe is the point.

        `DATASET_IMAGES` means "a header actually decoded", not "the manifest mentions
        images". The manifest is the attacker-supplied half of the input, so believing it
        would let a dataset declare a capability it does not have and take a detector down
        a path with nothing underneath it.
        """
        caps: set[Capability] = set()
        notes = list(self.notes)

        if any(s.width > 0 and s.height > 0 for s in self.samples):
            caps.add(Capability.DATASET_IMAGES)
        else:
            notes.append((Capability.DATASET_IMAGES,
                          "no sample produced decodable image dimensions"))

        if any(s.labels for s in self.samples):
            caps.add(Capability.DATASET_LABELS)
        else:
            notes.append((Capability.DATASET_LABELS, "no sample carries a label"))

        # "resolved on at least one Sample" — a column of None is not contributor metadata.
        if any(s.contributor is not None for s in self.samples):
            caps.add(Capability.DATASET_CONTRIBUTOR_META)
        else:
            notes.append((
                Capability.DATASET_CONTRIBUTOR_META,
                f"no contributor resolved — no {SIDECAR_NAME}, no contrib_* directory, "
                "and no source field in the manifest. The contributor rollup reports "
                "UNAVAILABLE rather than attributing every sample to one party."))
        return CapabilitySet(frozenset(caps), tuple(notes))


def probe_image(root: Path, file_name: str, max_pixels: int | None = None) -> tuple[Path, int, int]:
    """Resolve a manifest-supplied name to a real file and read its real dimensions.

    Both halves are safety controls, not conveniences. `safe_join` (S6) is what stops a
    `file_name` of `../../etc/passwd` escaping the dataset root — and it compares resolved
    paths rather than string prefixes, because `/data/root2` passes a `startswith` test
    against `/data/root`. `check_pixel_budget` (S7) reads the dimensions FROM THE HEADER
    and refuses before a decompression bomb is expanded into memory.
    """
    path = safe_join(root, file_name)
    w, h = (check_pixel_budget(path) if max_pixels is None
            else check_pixel_budget(path, max_pixels))
    return path, w, h


def _load_sidecar(root: Path) -> dict[str, str]:
    """Tier 1 — `contributors.yaml`, a flat `name-or-stem: contributor` mapping.

    The format is decided here because no vault document defines one; it is deliberately
    the smallest thing that answers the question. Parsed with `yaml.safe_load` — the
    dataset is untrusted, and `yaml.load` on an untrusted file constructs arbitrary Python
    objects, which is S2's problem wearing a different extension.
    """
    p = root / SIDECAR_NAME
    if not p.exists():
        return {}
    try:
        import yaml
        raw = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, str) and v:
            out[str(k)] = v
            out[Path(str(k)).name] = v
            out[Path(str(k)).stem] = v
    return out


def _from_directory(root: Path, path: Path) -> str | None:
    """Tier 2 — a path component that declares itself a contributor folder."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    for part in rel.parts[:-1]:
        low = part.lower()
        for pref in CONTRIB_DIR_PREFIXES:
            if low.startswith(pref) and len(part) > len(pref):
                return part[len(pref):]
    return None


def _clean_grouping_value(raw: str) -> str | None:
    """Make supplier-controlled text safe to carry as a `group_value`, or None if nothing is left.

    Control characters (category Cc), format characters (Cf, which includes bidi overrides and
    zero-width spaces), unassigned and private-use code points, and the line and paragraph
    separators are REPLACED with `_` rather than dropped: replacing keeps two different names
    different, and the value stays a readable plain string. Then trimmed and capped."""
    cleaned = "".join(
        "_" if unicodedata.category(c)[0] == "C" or c in "  " else c for c in raw)
    return cleaned.strip()[:GROUPING_MAX_LEN].rstrip() or None


def _from_prefixed_component(part: str, prefixes: tuple[str, ...]) -> str | None:
    """The cleaned remainder of one path component if it starts with one of `prefixes`."""
    for pref in prefixes:
        if part[:len(pref)].lower() == pref and len(part) > len(pref):
            return _clean_grouping_value(part[len(pref):])
    return None


def resolve_grouping(
    root: Path, path: Path, declared_batch: object = None,
) -> tuple[str | None, str | None]:
    """`(batch, source)` for one file: see the module docstring for the convention.

    `declared_batch` is a format's own batch field (COCO `images[].batch`); a string or a
    number counts (a number becomes its string), anything else, `bool` included, does not, and
    it beats a directory. Components are read from the file outwards, so the nearest match wins
    and a component whose remainder is empty after cleaning is passed over for the next one out.
    """
    batch: str | None = None
    if isinstance(declared_batch, str):
        batch = _clean_grouping_value(declared_batch)
    elif isinstance(declared_batch, int | float) and not isinstance(declared_batch, bool):
        batch = _clean_grouping_value(str(declared_batch))

    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return batch, None
    source: str | None = None
    for part in reversed(rel.parts[:-1]):
        if batch is None:
            batch = _from_prefixed_component(part, BATCH_DIR_PREFIXES)
        if source is None:
            source = _from_prefixed_component(part, SOURCE_DIR_PREFIXES)
    return batch, source


def resolve_contributor(
    root: Path, path: Path, sidecar: dict[str, str], declared: str | None,
) -> tuple[str | None, ContributorSource]:
    """The five-tier precedence, returning WHICH TIER answered alongside the answer.

    The tier is not bookkeeping. A contributor flagged from a signed sidecar and one
    inferred from an EXIF camera serial are not the same claim — the first is auditable
    evidence, the second a derived proxy the report must label a hypothesis — and the party
    being flagged is likely another unit. The spec already required the report to say which
    level was used; `contributor_source` is what lets it.

    Tier 4 (EXIF camera-serial clustering) is not implemented here: clustering is a
    dataset-wide operation, not a per-sample lookup, and it belongs with the Module A
    detector that does the clustering. Its absence is honest — it falls through to tier 5,
    which reports UNAVAILABLE rather than guessing.
    """
    for key in (str(path), path.name, path.stem):
        if key in sidecar:
            return sidecar[key], ContributorSource.SIDECAR

    from_dir = _from_directory(root, path)
    if from_dir:
        return from_dir, ContributorSource.DIRECTORY

    if declared:
        return declared, ContributorSource.FORMAT_FIELD

    return None, ContributorSource.NONE


def build_categories(source_ids: list[Any], names: dict[Any, str]) -> tuple[list[Category], dict[Any, int]]:
    """Produce the canonical dense 0-based mapping, and keep the source id beside it.

    This is the half of COCO/YOLO conversion that is not arithmetic and is where the real
    bugs live. COCO `category_id` values are arbitrary integers and need not be
    contiguous; YOLO indices are dense and zero-based. The mapping is DATA the loader
    produces, not a constant — and it has to survive into the report, because a finding
    that says "class 3" means nothing if two loaders disagree about what 3 is.
    """
    ordered = sorted(source_ids, key=lambda s: (isinstance(s, str), s))
    cats = [Category(category_id=i, name=names.get(sid, str(sid)), source_id=sid)
            for i, sid in enumerate(ordered)]
    return cats, {c.source_id: c.category_id for c in cats}


def read_json_safely(path: Path) -> Any:
    """S5 before `json.load`, never after.

    The caps have to be checked WITHOUT building the object, which is the whole point: a
    2000-deep nesting bomb raises `RecursionError` inside `json.load` itself, so a checker
    that parses first has already lost. `check_json_safety` streams the raw bytes through a
    state machine instead.
    """
    from cva.loaders.safety import check_json_safety

    check_json_safety(path)
    return json.loads(path.read_text())
