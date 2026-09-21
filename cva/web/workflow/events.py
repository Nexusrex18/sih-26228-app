"""`analyst_event` request builders and validation — the closed enums live here (plan §7.3).

**Why these enums are mirrored rather than imported.** Module C's `provenance/seal/records.py`
holds the authoritative `ANALYST_ACTIONS`, `REASON_CODES`, `TARGET_TYPES` and `DISPOSITIONS`,
and its `_v_analyst` validator is what actually decides whether a record may exist. This
module restates them because **`cva.web` must never import `cva.provenance`** (plan D-E3,
§5.2: the web process holds no key and no ledger handle; CI invariant 5 makes the import a
build failure). "One place" is preserved by a conformance test —
`tests/web/test_enum_conformance.py` imports both sides and asserts the tuples are equal — so
a drift is a red test rather than a code an analyst can type and a ledger will not store.

**Percent-encoding.** Module C's wire format is printable ASCII `0x20-0x7E` only, integers
only, no floats (plan §3.4). Nothing in `provenance.seal` encodes for you. So:

  * `encode_text()` is applied by the **request builder** here and again, defensively, by
    ledgerd — encoding an already-encoded string is not idempotent, so ledgerd encodes only
    when the value is not already pure-ASCII-safe; see `ledgerd/protocol.py`.
  * `decode_text()` is applied **only at render time**. Nothing stores a decoded string.

The round trip is tested with real non-ASCII prose (plan test 10.4).
"""
from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote

# --- the closed enums (mirrored from provenance/seal/records.py; see the module docstring) ---

ANALYST_ACTIONS = ("assign", "acknowledge", "override", "approve", "quarantine", "release")
DISPOSITIONS = ("accept", "review", "quarantine")
TARGET_TYPES = ("sample", "contributor", "batch", "model", "record", "dataset")

#: General-purpose codes, usable on any finding.
GENERAL_REASON_CODES = ("quality_issue", "known_benign", "insufficient_evidence",
                        "accepted_risk", "superseded_by_rescan", "other_with_justification")
#: D-E8: on a `prov.*` finding ONLY these are offered. A deterministic failure is arithmetic;
#: "the quality was poor" is not a statement anyone can make about a hash that did not match.
PROV_REASON_CODES = ("known_restore", "test_data", "superseded_ledger",
                     "false_positive_confirmed")
REASON_CODES = GENERAL_REASON_CODES + PROV_REASON_CODES

#: D-E8: `false_positive_confirmed` on a `prov.*` finding is the dishonest case. It is allowed
#: — some are genuinely wrong — and it is made loud: the event carries `audit_flag` in its
#: justification prefix and the timeline renders it in the audit-flagged style.
AUDIT_FLAGGED_REASON_CODES = ("false_positive_confirmed",)
AUDIT_FLAG_PREFIX = "[audit_flag] "

#: Ordering for "raises" vs "lowers" (D-E7). `accept` is the weakest claim about an asset and
#: `quarantine` the strongest, so an override TOWARDS quarantine is a raise and takes effect
#: immediately; an override away from it needs a second person.
DISPOSITION_RANK = {"accept": 0, "review": 1, "quarantine": 2}

#: Target-level actions act on an asset, not on one finding (§5.4).
TARGET_LEVEL_ACTIONS = ("quarantine", "release")

_ROLES = ("viewer", "analyst", "approver", "admin")

#: `_ident` in Module C's validator: `[A-Za-z0-9._:-]{1,64}`. Enforced at ACCOUNT CREATION
#: too (`accounts.py`), because the first analyst event from a user called "a sharma" would
#: otherwise be refused by the ledger at the worst possible moment.
ACTOR_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,64}")
#: `_text(..., hi=200)`, printable ASCII. A target_ref out of the audited material may be
#: anything at all, so it is percent-encoded like the justification.
_TARGET_REF_MAX = 200
_JUSTIFICATION_MAX = 8192
FINDING_ID_RE = re.compile(r"[0-9a-f]{16}")
REQUEST_ID_RE = re.compile(r"[0-9a-f]{32}")
SCAN_ID_RE = re.compile(r"s-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{4}")

#: Nothing outside this set may be percent-decoded back into the wire form. `quote` with a
#: `safe` of "" encodes every reserved character, so a decoded string can never re-parse as
#: structure.
_QUOTE_SAFE = ""


class EventError(ValueError):
    """A request the UI must refuse before it ever reaches ledgerd.

    `code` is the machine-readable reason the browser is shown (plan §5.3): the UI never
    reports a generic failure for something it decided itself.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# --- text on the wire ---------------------------------------------------------------------

def encode_text(s: str) -> str:
    """Percent-encode to printable ASCII for the ledger (§3.4)."""
    return quote(s, safe=_QUOTE_SAFE, encoding="utf-8")


def decode_text(s: str) -> str:
    """Decode for DISPLAY only. Jinja autoescape makes the result inert in HTML (S1)."""
    return unquote(s, encoding="utf-8", errors="replace")


def is_wire_safe(s: str) -> bool:
    """True when `s` is a CANONICAL percent-encoding: what the ledger accepts, exactly.

    Two conditions, and the second is the one that matters. Printable ASCII alone is not
    enough: a raw `seq:412` is printable ASCII and is *not* what `encode_text` produces, so
    accepting it would let two spellings of one value into the ledger. Requiring
    `encode(decode(s)) == s` pins the encoding to one form, which is the same malleability
    argument Module C makes for lowercase hex.

    Used by `cva-ledgerd` to REFUSE a badly encoded request, never to repair one: re-encoding
    is not idempotent (`%20` -> `%2520`), so a guess would corrupt exactly the analyst prose
    the record exists to preserve.
    """
    if not s:
        return True
    if not s.isascii() or any(not (0x20 <= ord(c) <= 0x7e) for c in s):
        return False
    return encode_text(decode_text(s)) == s


def synthetic_finding_id(target_type: str, target_ref: str) -> str:
    """The `finding_id` for a TARGET-LEVEL action that cites no finding.

    Module C's `_v_analyst` requires `finding_id` on every `analyst_event`, including a
    `quarantine`/`release` of a contributor or a batch — which has no finding of its own
    unless D4 happened to emit one. Padding it with zeros, or with the empty string, would
    make two different contributors share a key in the fold.

    So it is derived the same shape as Backend's own `finding_id`
    (`sha256(...)[:16]`, schema `^[0-9a-f]{16}$`) over the pair that identifies the target,
    with a domain prefix so it can never collide with a real detector finding. Callers that
    DO have a finding pass it instead; `target_event()` prefers the real one.
    """
    h = hashlib.sha256(b"cva-analyst-target\x00" + target_type.encode()
                       + b"\x00" + target_ref.encode()).hexdigest()
    return h[:16]


def new_request_id() -> str:
    """32 lowercase hex from a CSPRNG (plan §7.3, idempotency)."""
    return secrets.token_hex(16)


# --- the request ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalystEventRequest:
    """One typed, validated, wire-ready request. The web process builds it and ASKS.

    It never mutates anything; `cva-ledgerd` validates it again, independently, against the
    role table and the folded ledger before it signs (plan §5.1, trust rule).
    """

    body: Mapping[str, Any]

    @property
    def action(self) -> str:
        return str(self.body["action"])

    @property
    def finding_id(self) -> str:
        return str(self.body["finding_id"])

    def as_wire(self) -> dict[str, Any]:
        return dict(self.body)


def _require_enum(value: Any, allowed: tuple[str, ...], field: str) -> str:
    if value not in allowed:
        raise EventError("invalid_field",
                         f"{field} must be one of {list(allowed)}, got {value!r}")
    return str(value)


def _require_match(value: Any, pattern: re.Pattern[str], field: str, what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise EventError("invalid_field", f"{field} must be {what}, got {value!r}")
    return value


def _require_seq(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EventError("invalid_field", f"{field} must be a non-negative integer, got {value!r}")
    return value


def check_justification(text: str, reason_code: str, *, min_chars: int, min_words: int,
                        min_chars_other: int) -> str:
    """Plan §7.3: mandatory, trimmed, long enough, and **made of words**.

    "A string of punctuation is rejected" is not pedantry: `..............................`
    clears a 30-character minimum and records nothing. The check that matters is the word
    count, and the length floor is what stops five one-letter words.
    """
    trimmed = text.strip()
    if not trimmed:
        raise EventError("justification_required",
                         "An override records that a human judged a machine finding "
                         "acceptable. Without a reason in their own words, the record says "
                         "only that someone clicked.")
    floor = min_chars_other if reason_code == "other_with_justification" else min_chars
    if len(trimmed) < floor:
        raise EventError(
            "justification_too_short",
            f"at least {floor} characters are required"
            + (" for `other_with_justification`, which names no cause of its own"
               if reason_code == "other_with_justification" else "")
            + f"; this is {len(trimmed)}")
    words = [w for w in re.split(r"\s+", trimmed) if re.search(r"[^\W_]", w, re.UNICODE)]
    if len(words) < min_words:
        raise EventError(
            "justification_not_prose",
            f"at least {min_words} words are required; this has {len(words)}. "
            "Punctuation and repeated characters clear a length floor without recording "
            "anything a later auditor can read.")
    if len(encode_text(trimmed)) > _JUSTIFICATION_MAX:
        raise EventError("justification_too_long",
                         f"the percent-encoded justification exceeds {_JUSTIFICATION_MAX} "
                         "bytes; shorten it or attach the detail to the finding")
    return trimmed


def allowed_reason_codes(is_prov_finding: bool) -> tuple[str, ...]:
    """D-E8. On a `prov.*` finding the general codes are not offered at all."""
    return PROV_REASON_CODES if is_prov_finding else GENERAL_REASON_CODES


def build(*, actor_id: str, role: str, action: str, scan_id: str, target_type: str,
          target_ref: str, finding_id: str, justification: str = "",
          new_disposition: str | None = None, reason_code: str | None = None,
          assignee: str | None = None, refs_seq: int | None = None,
          expected_prev_seq: int | None = None, request_id: str | None = None,
          is_prov_finding: bool = False, min_chars: int = 30, min_words: int = 5,
          min_chars_other: int = 80) -> AnalystEventRequest:
    """Validate and assemble the `analyst` section Module C's `_v_analyst` will re-check.

    Everything here is refused *before* the socket, so the browser gets a reason it can act
    on rather than a generic `not_permitted` from the daemon. The daemon repeats every one of
    these checks — this is a usability layer over an independent gate, never a substitute
    for it (plan §5.1).
    """
    body: dict[str, Any] = {
        "actor_id": _require_match(actor_id, ACTOR_ID_RE, "actor_id",
                                   "an identifier [A-Za-z0-9._:-]{1,64}"),
        "role": _require_enum(role, _ROLES, "role"),
        "action": _require_enum(action, ANALYST_ACTIONS, "action"),
        "scan_id": _require_match(scan_id, SCAN_ID_RE, "scan_id",
                                  "a scan id s-YYYY-MM-DD-NNNN"),
        "target_type": _require_enum(target_type, TARGET_TYPES, "target_type"),
        "finding_id": _require_match(finding_id, FINDING_ID_RE, "finding_id",
                                     "16 lowercase hex characters"),
        "request_id": _require_match(request_id or new_request_id(), REQUEST_ID_RE,
                                     "request_id", "32 lowercase hex characters"),
    }

    if not isinstance(target_ref, str) or not target_ref:
        raise EventError("invalid_field", "target_ref must be a non-empty string")
    encoded_ref = encode_text(target_ref)
    if len(encoded_ref) > _TARGET_REF_MAX:
        raise EventError(
            "invalid_field",
            f"target_ref is {len(encoded_ref)} bytes once percent-encoded, over the "
            f"{_TARGET_REF_MAX}-byte ledger limit. The asset is identified by "
            "finding_id in the fold; a name this long belongs in the justification.")
    body["target_ref"] = encoded_ref

    if action == "override":
        body["new_disposition"] = _require_enum(new_disposition, DISPOSITIONS,
                                                "new_disposition")
        offered = allowed_reason_codes(is_prov_finding)
        if reason_code not in offered:
            raise EventError(
                "invalid_reason_code",
                f"reason_code must be one of {list(offered)} for "
                + ("a prov.* finding — a deterministic failure is arithmetic, so the "
                   "general codes do not describe anything that could be true of it"
                   if is_prov_finding else "a detector finding")
                + f"; got {reason_code!r}")
        body["reason_code"] = reason_code
        checked = check_justification(justification, str(reason_code), min_chars=min_chars,
                                      min_words=min_words, min_chars_other=min_chars_other)
        if reason_code in AUDIT_FLAGGED_REASON_CODES and is_prov_finding:
            checked = AUDIT_FLAG_PREFIX + checked
        body["justification"] = encode_text(checked)
        body["expected_prev_seq"] = _require_seq(
            expected_prev_seq if expected_prev_seq is not None else -1, "expected_prev_seq")
    elif action == "approve":
        body["refs_seq"] = _require_seq(refs_seq if refs_seq is not None else -1, "refs_seq")
        body["justification"] = encode_text(justification.strip())
    elif action == "assign":
        body["assignee"] = _require_match(assignee, ACTOR_ID_RE, "assignee",
                                          "an identifier [A-Za-z0-9._:-]{1,64}")
        body["justification"] = encode_text(justification.strip())
    else:
        # acknowledge / quarantine / release. `release` LOWERS an asset's status, so it needs
        # the same prose an override does (D-E7); `quarantine` raises and does not.
        if action == "release":
            checked = check_justification(justification, reason_code or "",
                                          min_chars=min_chars, min_words=min_words,
                                          min_chars_other=min_chars_other)
            body["justification"] = encode_text(checked)
        else:
            body["justification"] = encode_text(justification.strip())

    if action in TARGET_LEVEL_ACTIONS and expected_prev_seq is not None:
        body["expected_prev_seq"] = _require_seq(expected_prev_seq, "expected_prev_seq")

    return AnalystEventRequest(body=body)


def target_event_finding_id(target_type: str, target_ref: str,
                            known_finding_id: str | None = None) -> str:
    """Pick the `finding_id` a target-level action cites: the real one when there is one."""
    if known_finding_id and FINDING_ID_RE.fullmatch(known_finding_id):
        return known_finding_id
    return synthetic_finding_id(target_type, target_ref)


def is_prov(detector_id: str | None) -> bool:
    """D-E8's predicate, in one place: a `prov.*` finding is a deterministic provenance check."""
    return bool(detector_id) and str(detector_id).startswith("prov.")


__all__ = [
    "ACTOR_ID_RE", "ANALYST_ACTIONS", "AUDIT_FLAGGED_REASON_CODES", "AUDIT_FLAG_PREFIX",
    "AnalystEventRequest", "DISPOSITIONS", "DISPOSITION_RANK", "EventError",
    "GENERAL_REASON_CODES", "PROV_REASON_CODES", "REASON_CODES", "TARGET_LEVEL_ACTIONS",
    "TARGET_TYPES", "allowed_reason_codes", "build", "check_justification", "decode_text",
    "encode_text", "is_prov", "is_wire_safe", "new_request_id", "synthetic_finding_id",
    "target_event_finding_id",
]
