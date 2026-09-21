"""The SDK the field pipeline calls (plan §7.7, §8; decisions D8, D10, D15, C-13, C-14).

    sealer = Sealer.open(ledger_path, key=FileKeyProvider(...), trust_root=...)

    model_ref  = sealer.register_model(id=..., weights_sha256=..., arch_hash=..., format="onnx")   # once per load
    config_ref = sealer.register_config(preprocess_spec=..., postprocess_spec=..., runtime=..., ...) # once per load

    binding = sealer.bind_input(buf, source_kind="encoded_file", dims=(w, h))    # hashes `buf`
    result  = model.run(binding.data)                                            # binding.data IS what was hashed
    receipt = sealer.commit(binding, model_ref, config_ref, output=raw, filtered=decisions)

The API makes it impossible to hash one buffer and infer on another without deliberately re-reading
from a path (13-C4). `input.sha256` covers the ENCODED source bytes, or — when no encoded bytes exist —
the post-preprocessing model-input tensor; the record says which (Post-Merge R8).

Failure policy (§8). The default is FAIL-CLOSED: if the ledger cannot be written, `commit()` raises
`LedgerUnavailable` and THE CALLER MUST NOT RELEASE THE INFERENCE OUTPUT. `guard()` makes that the path
of least resistance. Fail-open is an explicit opt-in that needs `allow_fail_open=True` AND a signing key
that can actually sign at startup (a marker must be signable); unsealed inferences are counted and their
input hashes spilled to a side file, and on recovery a signed `degraded_marker` is written into the
chain. A declared hole is evidence; an undeclared hole is a lie by omission.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .errors import KeyNotConfigured, LedgerUnavailable, SealError, SealMissing, TrustRootError
from .keys import KeyProvider, TrustRoot, load_trust_root
from .outputs import build_output_objects
from .payloads import ref_for
from .records import (
    PHASH_OMITTED_REASONS,
    SOURCE_KINDS,
    format_utc,
    genesis_prev_hash,
    validate_section,
)
from .store import SealedLedger

FAIL_MODES = ("fail_closed", "fail_open")
NONFINITE_MODES = ("seal_marker", "raise")


@dataclass(frozen=True)
class SealPolicy:
    """Runtime policy. Durability is NOT here: it is chosen once at `init` and stored in the ledger's
    `meta`, so the loss window printed in a verify report cannot disagree with how the ledger was written."""

    on_ledger_failure: str = "fail_closed"
    allow_fail_open: bool = False
    on_nonfinite: str = "seal_marker"

    def __post_init__(self) -> None:
        if self.on_ledger_failure not in FAIL_MODES:
            raise ValueError(f"on_ledger_failure must be one of {FAIL_MODES}")
        if self.on_nonfinite not in NONFINITE_MODES:
            raise ValueError(f"on_nonfinite must be one of {NONFINITE_MODES}")
        if self.on_ledger_failure == "fail_open" and not self.allow_fail_open:
            raise ValueError("fail_open needs allow_fail_open=True — silent fail-open is the worst outcome")


@dataclass(frozen=True)
class ModelRef:
    id: str
    weights_sha256: str
    arch_hash: str
    format: str

    def section(self) -> dict[str, str]:
        return {"id": self.id, "weights_sha256": self.weights_sha256, "format": self.format,
                "arch_hash": self.arch_hash}


@dataclass(frozen=True)
class ConfigRef:
    preprocess_hash: str
    preprocess_ref: str
    postprocess_hash: str
    runtime: str
    version_pins_hash: str
    code_commit: str

    def section(self) -> dict[str, str]:
        return {"preprocess_hash": self.preprocess_hash, "preprocess_ref": self.preprocess_ref,
                "postprocess_hash": self.postprocess_hash, "runtime": self.runtime,
                "version_pins_hash": self.version_pins_hash, "code_commit": self.code_commit}


@dataclass(frozen=True)
class InputBinding:
    """The input as hashed. `data` is the SAME immutable object that was hashed: run the model on it."""

    data: bytes
    sha256: str
    source_kind: str
    dims: tuple[int, int]
    phash: Mapping[str, str] | None
    phash_omitted_reason: str | None


@dataclass(frozen=True)
class Receipt:
    sealed: bool                   # False only under fail-open, when the ledger could not be written
    seq: int | None                # seq of the inference record
    record_hash: str | None
    tree_size_after: int | None
    durable: bool                  # fsynced? False under group_commit until the next flush, and when unsealed


@dataclass
class _Gap:
    first: str
    last: str
    count: int
    spill_ok: bool = True


class Sealer:
    """One Sealer per process. Thread-safe (serialised); the ledger lock is the multi-process guard."""

    def __init__(self, ledger: SealedLedger, key: KeyProvider, policy: SealPolicy, spill_path: str,
                 clock: Callable[[], datetime]) -> None:
        self._ledger = ledger
        self._key = key
        self.policy = policy
        self._spill = spill_path
        self._clock = clock
        self._lock = threading.RLock()
        self._gap: _Gap | None = None
        self._registered: set[tuple[bytes, bytes]] = {
            (canonical_bytes(r["model"]), canonical_bytes(r["config"])) for r in ledger.registered_models()}

    # -- opening ---------------------------------------------------------------------------------

    @classmethod
    def open(cls, ledger_path: str | os.PathLike[str], *, key: KeyProvider | None,
             trust_root: TrustRoot | str | os.PathLike[str],
             policy: SealPolicy | None = None, clock: Callable[[], datetime] | None = None,
             rng: Callable[[int], bytes] | None = None, background_flush: bool = True) -> Sealer:
        """Open an EXISTING ledger for sealing. No key -> `KeyNotConfigured`: there is no fallback, no
        auto-generate and no development mode. A ledger is created only by `SealedLedger.init_ledger`
        (`cva-seal init`)."""
        if key is None:
            raise KeyNotConfigured("no signing key supplied; create one explicitly with `cva-seal keygen`")
        policy = policy or SealPolicy()
        if policy.on_ledger_failure == "fail_open" and not _can_sign(key):
            raise SealError("fail_open needs a signing key that can produce a verifying signature at startup "
                            "(a degraded_marker must be signable)")
        tr = trust_root if isinstance(trust_root, TrustRoot) else load_trust_root(trust_root)
        clock = clock or (lambda: datetime.now(UTC))
        ledger = SealedLedger.open(ledger_path, key=key, clock=clock, rng=rng, background_flush=background_flush)
        try:
            # the trust root vouches for the GENESIS key; a later active key is vouched for by the rotation
            # chain, which the verifier checks — so requiring it in the trust root would forbid rotation
            if ledger.genesis_key_id not in tr.ledger_keys():
                raise TrustRootError(f"the ledger's genesis key {ledger.genesis_key_id[:16]}… is not a ledger key "
                                     "in the trust root")
            if genesis_prev_hash(ledger.deployment_manifest) != tr.deployment_manifest_hash:
                raise TrustRootError("the trust root was issued for a different deployment manifest than this ledger's")
            sealer = cls(ledger, key, policy, os.fspath(ledger_path) + ".spill", clock)
            sealer._recover_spill()
        except BaseException:
            ledger.close()
            raise
        return sealer

    def rotate_key(self, new_key: KeyProvider) -> None:
        """Rotate the ledger's signing key (plan §5.10). Refuses a key that cannot sign, since a rotation to a
        key that then fails leaves the ledger unwritable."""
        with self._lock:
            if not _can_sign(new_key):
                raise SealError("the incoming key cannot produce a verifying signature; not rotating to it")
            self._ledger.rotate_key(new_key)
            self._key = new_key

    # -- registration (once per model load — NEVER per inference) ---------------------------------

    def register_model(self, *, id: str, weights_sha256: str, arch_hash: str, format: str) -> ModelRef:  # noqa: A002
        """Record which model is in use. The weights digest is computed ONCE at load by the caller
        (~100 ms on 100 MB) — the SDK never reads weights and never hashes them per inference."""
        ref = ModelRef(id, weights_sha256, arch_hash, format)
        validate_section("model", ref.section())                # early, precise errors
        return ref

    def register_config(self, *, preprocess_spec: Mapping[str, Any], postprocess_spec: Mapping[str, Any],
                        runtime: str, version_pins_hash: str, code_commit: str) -> ConfigRef:
        """Store the QUANTISED preprocessing/postprocessing specs (you cannot re-run preprocessing from a
        digest, so the spec must exist somewhere — it is stored in the ledger's payload table, and
        `prov.recompute` reads it back by reference) and return
        the refs. Specs must already be integers (use quantise.q_e6): floats never enter hashed JSON."""
        pre = canonical_bytes(preprocess_spec, max_bytes=None)
        post = canonical_bytes(postprocess_spec, max_bytes=None)
        pre_ref, post_ref = self._ledger.put_payloads([pre, post])
        ref = ConfigRef(preprocess_hash=pre_ref[len("sha256:"):], preprocess_ref=pre_ref,
                        postprocess_hash=post_ref[len("sha256:"):], runtime=runtime,
                        version_pins_hash=version_pins_hash, code_commit=code_commit)
        validate_section("config", ref.section())
        return ref

    # -- per inference ---------------------------------------------------------------------------

    def bind_input(self, buf: bytes | bytearray | memoryview, *, source_kind: str,
                   dims: tuple[int, int] | list[int], phash: Mapping[str, str] | None = None,
                   phash_omitted_reason: str | None = None) -> InputBinding:
        """Hash the input and hand back the exact bytes hashed. A `memoryview`/`bytearray` is snapshotted
        into immutable `bytes` first, so nothing can mutate the buffer between hash and inference."""
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"source_kind must be one of {SOURCE_KINDS}")
        if len(dims) != 2 or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in dims):
            raise ValueError("dims must be two positive integers [width, height]")
        if phash is None:
            reason = phash_omitted_reason or "not_computed"
            if reason not in PHASH_OMITTED_REASONS:
                raise ValueError(f"phash_omitted_reason must be one of {PHASH_OMITTED_REASONS}")
        else:
            if phash_omitted_reason is not None:
                raise ValueError("phash_omitted_reason must be None when a phash is supplied")
            reason = None
        data = buf if isinstance(buf, bytes) else bytes(buf)
        return InputBinding(data, hashlib.sha256(data).hexdigest(), source_kind, (dims[0], dims[1]),
                            dict(phash) if phash else None, reason)

    def commit(self, binding: InputBinding, model: ModelRef, config: ConfigRef, *, output: Mapping[str, Any],
               filtered: Mapping[str, Any] | None = None) -> Receipt:
        """Seal one inference. `output` is the RAW model output; `filtered` the post-processed decisions
        (defaults to `output`). Both payloads are inserted in the SAME TRANSACTION as the record, so a record
        can never reference a payload that does not exist, and one fsync makes both durable."""
        raw_o, fine_o, coarse_o = build_output_objects(output, filtered, on_nonfinite=self.policy.on_nonfinite)
        raw_b = canonical_bytes(raw_o, max_bytes=None)
        fine_b = canonical_bytes(fine_o, max_bytes=None)
        coarse_b = canonical_bytes(coarse_o, max_bytes=None)
        out_section = {"jcs_sha256": hashlib.sha256(fine_b).hexdigest(),
                       "decision_sha256": hashlib.sha256(coarse_b).hexdigest(),
                       "raw_jcs_sha256": hashlib.sha256(raw_b).hexdigest(),
                       "payload_ref": ref_for(fine_b)}
        input_section: dict[str, Any] = {"sha256": binding.sha256, "source_kind": binding.source_kind,
                                         "phash": dict(binding.phash) if binding.phash else None,
                                         "dims": list(binding.dims)}
        if binding.phash is None:
            input_section["phash_omitted_reason"] = binding.phash_omitted_reason
        model_s, config_s = model.section(), config.section()
        reg_key = (canonical_bytes(model_s), canonical_bytes(config_s))
        with self._lock:
            try:
                items: list[tuple[str, Mapping[str, Any]]] = []
                marker_gap = self._gap
                spill_sha = self._spill_sha(marker_gap) if marker_gap else None
                if marker_gap is not None:
                    items.append(("degraded_marker", {"gap": self._gap_section(marker_gap, spill_sha or "unavailable")}))
                if reg_key not in self._registered:
                    items.append(("model_registration", {"model": model_s, "config": config_s}))
                items.append(("inference", {"input": input_section, "model": model_s, "config": config_s,
                                            "output": out_section}))
                written = self._ledger.append_many(items, payloads=[raw_b, fine_b])
            except LedgerUnavailable:
                if self.policy.on_ledger_failure == "fail_closed":
                    raise
                self._note_unsealed(binding)
                return Receipt(False, None, None, None, False)
            self._registered.add(reg_key)
            if marker_gap is not None:
                self._gap = None
                self._retire_spill(written[-1].seq)
            inf = next(w for w in written if w.type == "inference")
            return Receipt(True, inf.seq, inf.record_hash, self._ledger.size(), self._ledger.durable)

    def seal(self, buf: bytes | bytearray | memoryview, model: ModelRef, config: ConfigRef, *, output: Mapping[str, Any],
             filtered: Mapping[str, Any] | None = None, source_kind: str = "encoded_file",
             dims: tuple[int, int] | list[int], phash: Mapping[str, str] | None = None) -> Receipt:
        """One-shot convenience: bind_input + commit. Prefer the two-call form when the model must run
        on the bound bytes between the calls."""
        binding = self.bind_input(buf, source_kind=source_kind, dims=dims, phash=phash)
        return self.commit(binding, model, config, output=output, filtered=filtered)

    @contextmanager
    def guard(self) -> Iterator[_Guard]:
        """Wrap "run the model, then release the output" so that releasing without sealing is impossible
        to do by accident:

            with sealer.guard() as g:
                binding = g.bind_input(buf, ...)
                result = model.run(binding.data)
                g.commit(binding, model_ref, config_ref, output=result)
                release(result)          # only reached if the seal succeeded (or fail-open counted it)

        If the ledger is unwritable under fail-closed, `commit` raises and `release` never runs. If the
        block ends WITHOUT committing, `SealMissing` is raised — the code-level form of selective logging.
        """
        g = _Guard(self)
        yield g
        if not g.committed:
            raise SealMissing("guard() block ended without sealing the inference it wrapped")

    # -- fail-open bookkeeping (plan §8) -----------------------------------------------------------

    def _note_unsealed(self, binding: InputBinding) -> None:
        ts = format_utc(self._clock())
        gap = self._gap or _Gap(ts, ts, 0)
        gap.last, gap.count = ts, gap.count + 1
        self._gap = gap
        try:
            fd = os.open(self._spill, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, f"{ts} {binding.sha256}\n".encode("ascii"))
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            gap.spill_ok = False            # only the in-memory counters survive; the marker will say so

    def _spill_sha(self, gap: _Gap) -> str | None:
        if not gap.spill_ok:
            return None
        try:
            return hashlib.sha256(Path(self._spill).read_bytes()).hexdigest()
        except OSError:
            return None

    @staticmethod
    def _gap_section(gap: _Gap, spill_sha256: str) -> dict[str, Any]:
        return {"first_unsealed_utc": gap.first, "last_unsealed_utc": gap.last, "reported_count": gap.count,
                "reason": "ledger_unwritable", "spill_sha256": spill_sha256}

    def _retire_spill(self, seq: int) -> None:
        """Keep the spill file as evidence, renamed so it can never be mistaken for a live one."""
        try:
            if os.path.exists(self._spill):
                os.replace(self._spill, f"{self._spill}.sealed-{seq}")
        except OSError:
            pass

    def _recover_spill(self) -> None:
        """An orphan spill means a process died while failing open: declare the hole it left."""
        try:
            data = Path(self._spill).read_bytes()
        except FileNotFoundError:
            return
        if not data.strip():
            self._retire_spill(0)
            return
        sha = hashlib.sha256(data).hexdigest()
        if self._ledger.has_degraded_marker_for(sha):                # crashed after the marker, before the rename
            self._retire_spill(self._ledger.size())
            return
        stamps = [ln.split(" ", 1)[0] for ln in data.decode("ascii", "replace").splitlines() if ln.strip()]
        first = stamps[0] if stamps else format_utc(self._clock())
        last = stamps[-1] if stamps else first
        gap = _Gap(first, last, len(stamps))
        written = self._ledger.append_many([("degraded_marker", {"gap": self._gap_section(gap, sha)})])
        self._retire_spill(written[-1].seq)

    # -- passthroughs ----------------------------------------------------------------------------

    @property
    def ledger(self) -> SealedLedger:
        return self._ledger

    def capabilities(self) -> set[str]:
        return self._ledger.capabilities()

    def flush(self) -> None:
        self._ledger.flush()

    def close(self) -> None:
        self._ledger.close()

    def __enter__(self) -> Sealer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _Guard:
    def __init__(self, sealer: Sealer) -> None:
        self._s = sealer
        self.committed = False

    def bind_input(self, *a: Any, **kw: Any) -> InputBinding:
        return self._s.bind_input(*a, **kw)

    def commit(self, *a: Any, **kw: Any) -> Receipt:
        r = self._s.commit(*a, **kw)
        self.committed = True                # sealed, or (fail-open) counted as a declared gap
        return r

    def seal(self, *a: Any, **kw: Any) -> Receipt:
        r = self._s.seal(*a, **kw)
        self.committed = True
        return r


def _can_sign(key: KeyProvider) -> bool:
    fn = getattr(key, "self_test", None)
    return bool(fn()) if callable(fn) else False
