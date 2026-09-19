"""The DECLARED preprocessing spec — backend_plan.md §7.7 and §9.2.

Backend owes Module C's seal record two fields, `config.preprocess_hash` and
`config.preprocess_ref`: a hash of the resolved preprocessing, and a pointer to the STORED spec,
because you cannot re-run preprocessing from a digest and `prov.recompute` is the strongest check
in the system. Nothing in the vault said where the "resolved spec" comes from. It is DECLARED,
never inferred:

  * the operator writes `<model_path>.preprocess.json` beside the artefact, or passes
    `cva scan --preprocess <path>`;
  * it is NOT read out of the model file. ONNX `metadata_props` and a TorchScript archive carry
    nothing reliable about preprocessing, and the whole point of a declared spec is that
    `prov.recompute` re-runs exactly that.

The file is written by the model supplier, so it is untrusted. It goes through S5 (bounded size,
depth and item count, measured before the parser sees it) and is then validated IN CODE — this is
deliberately not a fourth schema file; §7.8's "three schemas" stands. Every rejection names the
field. A malformed spec is an error, never silently ignored: ignoring it would record "no
preprocessing declared" for a model whose operator DID declare one.

The accepted keys, and no others:

    mean         REQUIRED  non-empty list of finite numbers (|x| <= 1e6)
    std          REQUIRED  same length as `mean`, every value finite and > 0 (|x| <= 1e6)
    layout       REQUIRED  "CHW" or "HWC"
    dtype        REQUIRED  one of float16 / float32 / float64 / uint8
    value_range  optional  [lo, hi], finite, lo < hi: the range the pixel values are scaled to
                           before mean/std are applied
    input_shape  optional  per-sample [C, H, W] or [H, W, C] after preprocessing; when present,
                           its channel axis (per `layout`) must equal len(mean) == len(std)

Hash: `mean`, `std` and `value_range` are quantised with the normative
`floor(float64(x) * 1e6 + 0.5)` (`cva.core.quantise.q_e6` — NEVER `round()`), the result is
JCS-canonicalised, and sha256 of those bytes is `preprocess_hash`. The canonical bytes are what is
stored in the evidence store, so `preprocess_ref` — the bare content-addressed name — is
`<preprocess_hash>.json` by construction. The quantised form contains only integers and strings
with `[a-z0-9_]` keys, which is the profile the seal's canonicaliser accepts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from cva.core.quantise import canonical_bytes, q_e6
from cva.loaders.safety import UnsafeArtifact, check_json_safety

SIDECAR_SUFFIX = ".preprocess.json"

#: A preprocessing spec is a handful of numbers. These caps are far above any real one and far
#: below anything that could hurt: a spec over 64 KiB or 8 levels deep is not a spec.
MAX_SPEC_BYTES = 64 << 10
MAX_SPEC_DEPTH = 8
MAX_SPEC_ITEMS = 1_000

MAX_CHANNELS = 256
#: Keeps every quantised value (x * 1e6) far inside +/-(2**53 - 1), where JCS is exact.
MAX_ABS_VALUE = 1_000_000.0
MAX_DIM = 65_536

KNOWN_KEYS = ("mean", "std", "layout", "dtype", "value_range", "input_shape")
REQUIRED_KEYS = ("mean", "std", "layout", "dtype")
LAYOUTS = ("CHW", "HWC")
DTYPES = ("float16", "float32", "float64", "uint8")


class PreprocessSpecError(ValueError):
    """The declared preprocessing spec is unusable. The message names the field."""


def sidecar_path(model_path: str | Path) -> Path:
    """`<model_path>.preprocess.json`, beside the artefact (also for a directory model)."""
    p = Path(model_path)
    return p.with_name(p.name + SIDECAR_SUFFIX)


# --- loading -------------------------------------------------------------------------------

def _reject_constant(name: str) -> Any:
    raise ValueError(f"the JSON literal {name} is not a number")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def load_preprocess_spec(path: str | Path) -> dict[str, Any]:
    """Read, bound (S5), parse and validate a declared spec; return the NORMALISED spec
    (numbers as floats, shapes as ints) or raise `PreprocessSpecError`."""
    path = Path(path)
    if not path.is_file():
        raise PreprocessSpecError(f"{path}: preprocessing spec not found (not a file)")
    try:
        check_json_safety(path, max_bytes=MAX_SPEC_BYTES, max_depth=MAX_SPEC_DEPTH,
                          max_items=MAX_SPEC_ITEMS)
    except UnsafeArtifact as exc:
        raise PreprocessSpecError(f"{path}: {exc.reason} [{exc.control}]") from exc
    try:
        raw = json.loads(path.read_bytes().decode("utf-8"), parse_constant=_reject_constant,
                         object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PreprocessSpecError(f"{path}: not valid JSON ({exc})") from exc
    return validate_preprocess_spec(raw, source=str(path))


# --- validation ----------------------------------------------------------------------------

def _number(v: Any, field: str, source: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise PreprocessSpecError(
            f"{source}: {field} must be a number, got {type(v).__name__} ({v!r})")
    try:
        f = float(v)
    except OverflowError as exc:            # an int too large for a double
        raise PreprocessSpecError(f"{source}: {field} is out of range") from exc
    if not math.isfinite(f):
        raise PreprocessSpecError(f"{source}: {field} must be finite, got {v!r}")
    if abs(f) > MAX_ABS_VALUE:
        raise PreprocessSpecError(
            f"{source}: {field} = {f!r} is outside +/-{MAX_ABS_VALUE:g}")
    return f


def _numbers(v: Any, field: str, source: str, *, min_len: int, max_len: int) -> list[float]:
    if not isinstance(v, list):
        raise PreprocessSpecError(f"{source}: {field} must be a list of numbers")
    if not min_len <= len(v) <= max_len:
        raise PreprocessSpecError(
            f"{source}: {field} must have between {min_len} and {max_len} entries, "
            f"got {len(v)}")
    return [_number(x, f"{field}[{i}]", source) for i, x in enumerate(v)]


def validate_preprocess_spec(raw: Any, source: str = "preprocess spec") -> dict[str, Any]:
    """Validate a parsed spec and return it normalised. Idempotent on its own output."""
    if not isinstance(raw, dict):
        raise PreprocessSpecError(f"{source}: the top level must be a JSON object")
    unknown = sorted(str(k) for k in raw if k not in KNOWN_KEYS)
    if unknown:
        raise PreprocessSpecError(
            f"{source}: unknown key(s) {unknown}; the accepted keys are {list(KNOWN_KEYS)}")
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise PreprocessSpecError(f"{source}: required key {key!r} is missing")

    mean = _numbers(raw["mean"], "mean", source, min_len=1, max_len=MAX_CHANNELS)
    std = _numbers(raw["std"], "std", source, min_len=1, max_len=MAX_CHANNELS)
    if len(mean) != len(std):
        raise PreprocessSpecError(
            f"{source}: mean has {len(mean)} entries but std has {len(std)}")
    for i, s in enumerate(std):
        if not s > 0.0:
            raise PreprocessSpecError(f"{source}: std[{i}] must be > 0, got {s!r}")

    layout = raw["layout"]
    if not isinstance(layout, str) or layout not in LAYOUTS:
        raise PreprocessSpecError(f"{source}: layout must be one of {list(LAYOUTS)}, got {layout!r}")
    dtype = raw["dtype"]
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise PreprocessSpecError(f"{source}: dtype must be one of {list(DTYPES)}, got {dtype!r}")

    out: dict[str, Any] = {"mean": mean, "std": std, "layout": layout, "dtype": dtype}

    if "value_range" in raw:
        lo, hi = _numbers(raw["value_range"], "value_range", source, min_len=2, max_len=2)
        if not lo < hi:
            raise PreprocessSpecError(
                f"{source}: value_range must have lo < hi, got [{lo!r}, {hi!r}]")
        out["value_range"] = [lo, hi]

    if "input_shape" in raw:
        shape = raw["input_shape"]
        if not isinstance(shape, list) or len(shape) != 3:
            raise PreprocessSpecError(
                f"{source}: input_shape must be a list of exactly 3 integers "
                f"([C, H, W] or [H, W, C] per layout)")
        dims: list[int] = []
        for i, d in enumerate(shape):
            if isinstance(d, bool) or not isinstance(d, int):
                raise PreprocessSpecError(
                    f"{source}: input_shape[{i}] must be an integer, got {d!r}")
            if not 1 <= d <= MAX_DIM:
                raise PreprocessSpecError(
                    f"{source}: input_shape[{i}] = {d} is outside 1..{MAX_DIM}")
            dims.append(d)
        channels = dims[0] if layout == "CHW" else dims[-1]
        if not len(mean) == len(std) == channels:
            raise PreprocessSpecError(
                f"{source}: input_shape {dims} with layout {layout} has {channels} channels "
                f"but mean has {len(mean)} and std has {len(std)} entries")
        out["input_shape"] = dims
    return out


# --- the hash ------------------------------------------------------------------------------

def _no_floats(o: Any, where: str = "$") -> None:
    if isinstance(o, float):
        raise TypeError(f"an unquantised float reached the hashed form at {where}")
    if isinstance(o, dict):
        for k, v in o.items():
            _no_floats(v, f"{where}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            _no_floats(v, f"{where}[{i}]")


def quantise_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """The hashed form: `mean`/`std`/`value_range` as `floor(x * 1e6 + 0.5)` integers, built
    field by field so no float can slip through a generic walk. `quantisation` is recorded so
    whoever reads the STORED bytes back (`prov.recompute`) knows the scale."""
    v = validate_preprocess_spec(spec)          # normalises 1 vs 1.0 and re-checks everything
    q: dict[str, Any] = {
        "quantisation": "e6",
        "mean": [q_e6(x) for x in v["mean"]],
        "std": [q_e6(x) for x in v["std"]],
        "layout": v["layout"],
        "dtype": v["dtype"],
    }
    if "value_range" in v:
        q["value_range"] = [q_e6(x) for x in v["value_range"]]
    if "input_shape" in v:
        q["input_shape"] = list(v["input_shape"])
    _no_floats(q)
    return q


def preprocess_digest(spec: dict[str, Any]) -> tuple[str, bytes]:
    """`(preprocess_hash, canonical_bytes)` of the QUANTISED spec: sha256 of the JCS bytes."""
    blob = canonical_bytes(quantise_spec(spec))
    return hashlib.sha256(blob).hexdigest(), blob


def preprocess_ref_of(preprocess_hash: str) -> str:
    """The bare content-addressed name the evidence store gives these bytes."""
    return f"{preprocess_hash}.json"


# --- attaching to a handle -------------------------------------------------------------------

def attach_preprocess(handle: Any, model_path: str | Path | None = None,
                      explicit: str | Path | None = None) -> bool:
    """Attach `preprocess_hash`, `preprocess_ref` and `preprocess_spec_bytes` to `handle`.

    `explicit` (`--preprocess`) wins; otherwise the sidecar beside `model_path`, if there is
    one. Returns whether a spec was attached. With no spec NOTHING is attached, so consumers
    read the absence with `getattr(handle, "preprocess_hash", None)` and the report says so
    rather than recording an empty string.

    These are plain attributes, not members of the frozen `ModelHandle` Protocol: the spec is
    a fact about how the operator feeds the model, not something a loader can probe.
    """
    if explicit is not None:
        spec_path: Path | None = Path(explicit)
    elif model_path is not None and sidecar_path(model_path).is_file():
        spec_path = sidecar_path(model_path)
    else:
        spec_path = None
    if spec_path is None:
        return False
    digest, blob = preprocess_digest(load_preprocess_spec(spec_path))
    handle.preprocess_hash = digest
    handle.preprocess_ref = preprocess_ref_of(digest)
    handle.preprocess_spec_bytes = blob
    return True
