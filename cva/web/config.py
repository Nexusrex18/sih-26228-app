"""Validated configuration. **An unknown key is a load error, not a warning** (plan N0).

The reasoning is the one `profiles/default.yaml` already states for thresholds: a typo in a
setting name that silently falls back to a default is indistinguishable from a configured
system. Here it is worse than indistinguishable — `bind_host: 0.0.0.0` misspelt as
`bind_hosts` would leave the dashboard on loopback while the operator believes it is on the
LAN, or the reverse.

Every value is typed and range-checked at load. Nothing reads `os.environ` at request time.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised at load. Never caught to fall back on a default — that is the whole point."""


@dataclass(frozen=True)
class WebConfig:
    # --- where the artefacts are -------------------------------------------------------
    #: `<out_dir>` of `cva scan`: holds `<scan_id>/report.json` and the SHARED `evidence/`.
    reports_dir: Path = Path("artifacts/reports")
    #: Derived, rebuildable cache. Delete it and it rebuilds; nothing authoritative lives here.
    index_db: Path = Path("var/index.db")
    #: Local accounts (users, roles, lockout state).
    accounts_db: Path = Path("var/accounts.db")
    #: The ONLY channel to the ledger. The web process never opens the ledger itself.
    ledgerd_socket: Path = Path("/run/cva/ledgerd.sock")
    #: The trust root `cva-seal verify` is run against. Absent -> verification UNAVAILABLE,
    #: stated as such; the workflow is then read-only, because a fold we cannot verify is
    #: not a record we may act on (plan §5.4, ledger-trust banner).
    trust_root: Path | None = None
    #: Passed to `cva-seal verify`. Absent -> the bridge reports UNAVAILABLE with that reason.
    ledger_path: Path | None = None

    # --- binding (S9) -------------------------------------------------------------------
    bind_host: str = "127.0.0.1"
    bind_port: int = 8713
    #: Required, with a certificate, before `bind_host` may leave loopback: no plaintext
    #: credentials on a shared network.
    tls_certfile: Path | None = None
    tls_keyfile: Path | None = None

    # --- session and lockout (S6, S8) ----------------------------------------------------
    session_idle_timeout_s: int = 900          # 15 minutes
    lockout_threshold: int = 5
    lockout_base_s: int = 2                    # exponential: base * 2**(failures - threshold)
    lockout_max_s: int = 3600

    # --- workflow policy (D-E7, §7.3) -----------------------------------------------------
    min_justification_chars: int = 30
    min_justification_words: int = 5
    #: `other_with_justification` asks for more, because it names no cause.
    min_justification_chars_other: int = 80
    #: Four-eyes. Policy, not contract (O3): the team may set this to False without the fold
    #: changing shape — `approve` events are still recorded, they simply stop being required.
    four_eyes: bool = True

    # --- rendering and limits (§9, S12) ----------------------------------------------------
    findings_per_page: int = 100
    max_request_bytes: int = 65536             # 64 KiB; justifications are text
    rate_limit_per_minute: int = 120
    #: Wall clock for an allowlisted subprocess (`cva-seal verify`, `cva remediate`).
    subprocess_timeout_s: int = 300

    # --- provenance of this config --------------------------------------------------------
    source: str = field(default="<defaults>", compare=False)

    # ---------------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.bind_host != "127.0.0.1" and not (self.tls_certfile and self.tls_keyfile):
            raise ConfigError(
                "S9: bind_host is not loopback, so TLS is REQUIRED — set tls_certfile and "
                "tls_keyfile. A dashboard reachable on a LAN over plain HTTP puts analyst "
                "passwords and session cookies on the wire of a network whose other hosts "
                "are, by this tool's own premise, not trusted.")
        for name, lo in (("bind_port", 1), ("session_idle_timeout_s", 30),
                         ("lockout_threshold", 1), ("lockout_base_s", 1), ("lockout_max_s", 1),
                         ("min_justification_chars", 1), ("min_justification_words", 1),
                         ("findings_per_page", 1), ("max_request_bytes", 1024),
                         ("rate_limit_per_minute", 1), ("subprocess_timeout_s", 1)):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < lo:
                raise ConfigError(f"{name} must be an integer >= {lo}, got {v!r}")
        if self.bind_port > 65535:
            raise ConfigError(f"bind_port must be <= 65535, got {self.bind_port}")
        if self.min_justification_chars_other < self.min_justification_chars:
            raise ConfigError(
                "min_justification_chars_other must not be below min_justification_chars: "
                "`other_with_justification` names no cause, so it asks for MORE prose, "
                "never less.")


_PATHS = frozenset({"reports_dir", "index_db", "accounts_db", "ledgerd_socket", "trust_root",
                    "ledger_path", "tls_certfile", "tls_keyfile"})
_BOOLS = frozenset({"four_eyes"})
_STRS = frozenset({"bind_host", "source"})
_INTS = frozenset({f.name for f in fields(WebConfig)}) - _PATHS - _BOOLS - _STRS


def _coerce(name: str, raw: Any) -> Any:
    if name in _PATHS:
        if raw is None:
            return None
        if not isinstance(raw, (str, Path)) or (isinstance(raw, str) and not raw.strip()):
            raise ConfigError(f"{name}: expected a non-empty path, got {raw!r}")
        return Path(raw).expanduser()
    if name in _BOOLS:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in ("true", "false", "yes", "no", "1", "0"):
            return raw.strip().lower() in ("true", "yes", "1")
        raise ConfigError(f"{name}: expected a boolean, got {raw!r}")
    if name in _INTS:
        if isinstance(raw, bool):
            raise ConfigError(f"{name}: expected an integer, got a boolean")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
            return int(raw.strip())
        raise ConfigError(f"{name}: expected an integer, got {raw!r}")
    if not isinstance(raw, str):
        raise ConfigError(f"{name}: expected a string, got {raw!r}")
    return raw


def from_mapping(data: Mapping[str, Any], *, source: str = "<mapping>") -> WebConfig:
    """Build a config from an already-parsed mapping. Unknown keys raise."""
    known = {f.name for f in fields(WebConfig)} - {"source"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{source}: unknown configuration key(s) {unknown}. Known keys are "
            f"{sorted(known)}. A misspelt setting that falls back to a default is "
            f"indistinguishable from a configured system, so this is a load error.")
    kwargs = {k: _coerce(k, v) for k, v in data.items()}
    return WebConfig(source=source, **kwargs)


def load(path: str | os.PathLike[str] | None = None,
         environ: Mapping[str, str] | None = None) -> WebConfig:
    """Load from a YAML/JSON file, then apply `CVA_WEB_*` environment overrides.

    The environment is read ONCE, here. A setting that can change between two requests of
    the same process is a setting nobody can audit.
    """
    environ = os.environ if environ is None else environ
    data: dict[str, Any] = {}
    source = "<defaults>"
    path = path or environ.get("CVA_WEB_CONFIG")
    if path:
        p = Path(path)
        try:
            text = p.read_text()
        except OSError as e:
            raise ConfigError(f"{p}: {e.strerror}") from None
        source = str(p)
        data = _parse(text, source)
    prefix = "CVA_WEB_"
    env_used = False
    for key, value in sorted(environ.items()):
        if key.startswith(prefix) and key != "CVA_WEB_CONFIG":
            data[key[len(prefix):].lower()] = value
            env_used = True
    if env_used:
        source = f"{source} + environment"
    return from_mapping(data, source=source)


def _parse(text: str, source: str) -> dict[str, Any]:
    import json
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{source}: not valid JSON ({e})") from None
    else:
        try:
            import yaml
        except ModuleNotFoundError:                       # pragma: no cover - PyYAML is a core dep
            raise ConfigError(f"{source}: PyYAML is not installed and this is not JSON") from None
        try:
            loaded = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"{source}: not valid YAML ({e})") from None
    if not isinstance(loaded, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level, got "
                          f"{type(loaded).__name__}")
    return loaded
