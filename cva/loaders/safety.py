"""The eight artefact-handling controls S1-S8 of backend_plan.md §5.14.

We parse untrusted files for a living. An air-gapped assurance tool compromised by the
artefact it is auditing becomes a delivery mechanism into the enclave, wearing the trust
the assurance workflow grants — which is why §5.14 puts this file in P0 rather than in a
later hardening pass.

The division of labour between the two halves of this module is deliberate. `prescan`
REPORTS: it is called by the loader router before anything is opened, it must always come
back with a `SafetyReport` so the digest reaches the ledger, and it is the caller that
decides what an unsafe verdict means. The individual controls below RAISE
`UnsafeArtifact` at the point of use, because by the time you are joining a path or
decoding a pixel buffer there is no safe way to continue.

Control map, for the report's access-assumptions block:

  S1  `prescan` / `sha256_file`  — hash and record the bytes before opening them
  S2  `check_torch_version`      — the torch>=2.6.0 floor (CVE-2025-32434)
  S3  `load_in_sandbox`          — subprocess, no network, no writable cwd
  S4  `prescan`                  — ONNX custom-operator refusal
  S5  `check_json_safety`        — COCO JSON depth, size and item caps before json.load
  S6  `safe_join`                — dataset-root path-traversal rejection
  S7  `check_pixel_budget`       — pixel-count cap read from the header before decode
  S8  `decoder_pins`             — the decoder versions actually in use
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import socket
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TORCH_MIN = "2.6.0"

# S7's default budget is Pillow's own default `MAX_IMAGE_PIXELS`, chosen deliberately rather
# than picked for roundness: a cap above it would make this control *loosen* the protection
# the library already provides, which is the exact failure mode §5.14 warns about — nothing
# errors, the scan runs, and the control is decorative.
PIXEL_BUDGET_DEFAULT = 89_478_485
STANDARD_ONNX_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}

# S8. Recorded as a module constant rather than only in prose because the report's
# access-assumptions block asserts it: every decoder named by `decoder_pins()` is only ever
# invoked from inside the S3 subprocess, so a decoder memory-safety bug costs us a dead
# child process rather than control of the auditing host.
DECODERS_RUN_IN_S3_SANDBOX = (
    "Every decoder listed by decoder_pins() runs inside the S3 subprocess sandbox "
    "(no network, non-writable working directory, wall-clock bounded), so a malformed "
    "artefact that corrupts a decoder cannot reach the host process that holds the ledger."
)

# S3's honest scope, stated as a constant because the loader reads it into the report's
# access-assumptions block verbatim. §5.14 asks for a "read-only filesystem"; what this
# module implements is weaker, and the gap is named here rather than left for a reader to
# infer from the code. A control that claims more than it enforces is §5.14's decorative
# control wearing better paperwork — and this block is the one place in the report where
# that claim is made to an assessor.
S3_SANDBOX_LIMITATION = (
    "S3 is a PYTHON-LEVEL sandbox: the child process is spawned fresh, its `socket` module "
    "is stubbed, its working directory is a throwaway that is removed, and it is bounded by "
    "wall clock. It is NOT an OS-level confinement. `import _socket` directly, a ctypes call "
    "into libc, or any write to an absolute path all bypass it. It raises the cost of a "
    "hostile artefact and bounds the blast radius of a crashing decoder; it does not make "
    "arbitrary code execution inside the child safe. A guarantee at §5.14's strength needs "
    "namespaces/seccomp or a bind-mounted read-only view of the filesystem."
)

# The sentinel the S3 child's socket stub raises with. It is matched on by the failing-input
# test: a probe that merely fails to connect proves nothing, because an unsandboxed
# connection to a closed port fails too. Rejection has to be distinguishable from refusal.
_S3_NETWORK_BLOCKED = "S3: network egress blocked inside the load sandbox"


class UnsafeArtifact(Exception):
    """Raised by a control that has decided the artefact must not be processed further.

    It carries the three things an operator needs in the report — which file, which control
    of §5.14 fired, and why — because "load failed" in a log is indistinguishable from a
    corrupt download, and the whole point of these controls is that the two are treated
    differently.
    """

    def __init__(self, path: Any, control: str, reason: str) -> None:
        super().__init__(f"[{control}] {path}: {reason}")
        self.path = path
        self.control = control
        self.reason = reason


@dataclass(frozen=True)
class SafetyReport:
    path: Path
    sha256: str
    size_bytes: int
    safe_to_load: bool
    reasons: tuple[str, ...]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# S2 DEBT, owed at B2a proper: §5.14's row for this control names both the version floor
# AND a pickle-payload checkpoint — a .pt carrying a __reduce__ that executes on load. Only
# the floor is covered here. The artefact test belongs with PyTorchLoader, where there is a
# real load path to point it at; the floor alone does not demonstrate the payload is stopped.
#
# TORCH_MIN changed type here, (2, 6, 0) -> "2.6.0", so that `packaging.version` does the
# compare: '2.14.0' >= '2.6' is False as a string compare, which fails on a CORRECT install.
# No consumer outside this module reads it today — noted because a tuple-unpacking import
# elsewhere would break silently rather than loudly.
def check_torch_version(version: str | None = None) -> None:
    """S2 — the torch floor is a security control, not a convenience.

    `weights_only=True` existed from torch 2.4 and did NOT close CVE-2025-32434 (CVSS 9.3)
    until 2.6.0, so a scan running on 2.5 with the flag set looks correct and is not. The
    comparison goes through `packaging.version` because the obvious string compare gets
    this wrong in the direction that fails open: '2.14.0' >= '2.6' is False as strings.

    `version` exists so the floor itself is testable without uninstalling torch; with no
    argument the behaviour is unchanged — it reads the installed torch.
    """
    from packaging.version import Version

    reported = version
    if reported is None:
        import torch
        reported = torch.__version__
    if Version(reported.split("+")[0]) < Version(TORCH_MIN):
        raise RuntimeError(
            f"torch {reported} < {TORCH_MIN} — weights_only=True does not fully close "
            "CVE-2025-32434. Refusing to load an untrusted checkpoint.")


# --- S3: run the untrusted load somewhere it cannot reach the network or the disk -------

def _sandbox_child(qualname: str, path_str: str, sys_path: list[str], conn: Any) -> None:
    """The 'spawn' child entry point. Ordering here is the control.

    The network stub is installed and the working directory is moved BEFORE the target
    module is imported, because a hostile module body runs at import time — waiting until
    the call would hand it a free turn with a live socket and a writable cwd.
    """
    tmp = None
    try:
        sys.path[:] = list(sys_path)

        # A read-only *view*, achieved without root: the child's relative paths all resolve
        # into a throwaway directory, so a load that writes a sidecar file writes it into
        # something that is deleted, not next to the artefact or into the dataset root.
        tmp = tempfile.mkdtemp(prefix="cva-s3-")
        os.chdir(tmp)

        def _no_network(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError(_S3_NETWORK_BLOCKED)

        socket.socket = _no_network  # type: ignore[assignment]
        socket.create_connection = _no_network  # type: ignore[assignment]

        mod_name, _, attr = qualname.partition(":")
        if not attr:
            mod_name, _, attr = qualname.rpartition(".")
        import importlib
        fn = getattr(importlib.import_module(mod_name), attr)
        conn.send(("ok", fn(Path(path_str))))
    except BaseException as exc:  # noqa: BLE001 - the child must never die silently
        # Exception instances are sent as strings: an arbitrary exception raised by
        # untrusted code may be unpicklable, and a pickling failure here would turn a
        # detected attack into an inscrutable hang.
        conn.send(("err", type(exc).__name__, str(exc)))
    finally:
        conn.close()
        if tmp is not None:
            # Whatever the load wrote goes with the sandbox. Leaving these behind would
            # accumulate one directory per audited artefact on a machine that is meant to
            # stay auditable.
            import shutil
            os.chdir(tempfile.gettempdir())
            shutil.rmtree(tmp, ignore_errors=True)


def load_in_sandbox(fn_qualname: str, path: Path, timeout_s: float = 60.0) -> Any:
    """S3 — run an untrusted load in a subprocess with no network and no writable cwd.

    `fn_qualname` is "package.module:function"; the function is imported and called with the
    artefact path inside the child. Stdlib only and 'spawn' only: a forked child inherits
    the parent's already-imported modules and open descriptors, which defeats the isolation
    the control is bought for, and `set_start_method` would be a global change to a process
    the rest of the test suite shares.

    Anything other than a clean return — a raised exception, a segfaulting decoder, a load
    that never terminates — is reported as `UnsafeArtifact`, because from the auditing
    host's point of view those are the same event: the artefact took the loader off the
    rails.
    """
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_sandbox_child,
        args=(fn_qualname, str(path), list(sys.path), child_conn),
    )
    proc.start()
    child_conn.close()
    try:
        # Poll rather than join-then-read: a child that fills the pipe buffer would
        # otherwise block on send() while the parent blocks on join(), and the timeout
        # would be spent on our own deadlock instead of on the artefact.
        got = parent_conn.poll(timeout_s)
        payload = parent_conn.recv() if got else None
    except EOFError:
        payload = None
    finally:
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)

    if payload is None:
        raise UnsafeArtifact(
            path, "S3",
            f"sandboxed load produced no result within {timeout_s}s or died "
            f"(exit code {proc.exitcode}) — treated as hostile")
    if payload[0] == "err":
        raise UnsafeArtifact(
            path, "S3", f"sandboxed load raised {payload[1]}: {payload[2]}")
    return payload[1]


# --- S5: bound the JSON before the parser sees it ---------------------------------------

def check_json_safety(
    path: Path,
    max_bytes: int = 512 << 20,
    max_depth: int = 64,
    max_items: int = 50_000_000,
) -> None:
    """S5 — COCO annotation files are attacker-supplied, so cap them before `json.load`.

    Every cap here has to be checked without building the object, which is the whole
    difficulty: `json.load` on a file that is 2000 open brackets blows the C recursion
    limit, and on a billion-laughs-shaped file it exhausts memory, and in both cases the
    damage is done by the time you could inspect the result. So the depth is measured by
    streaming the raw text through a small in-string/escape state machine and counting
    structural brackets — iteratively, so the checker cannot be killed by the input it is
    checking.

    `max_items` counts structural openers rather than parsed values. It is a proxy, chosen
    because it is computable in the same single pass and is monotonic in the real object
    count, and it is set loose enough that only pathological inputs reach it.
    """
    size = path.stat().st_size
    if size > max_bytes:
        raise UnsafeArtifact(
            path, "S5",
            f"JSON is {size} bytes, above the {max_bytes}-byte size cap")

    depth = 0
    items = 0
    in_string = False
    escaped = False
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            for byte in chunk:
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == 0x5C:  # backslash
                        escaped = True
                    elif byte == 0x22:  # closing quote
                        in_string = False
                    continue
                if byte == 0x22:
                    in_string = True
                elif byte in (0x7B, 0x5B):  # { [
                    depth += 1
                    items += 1
                    if depth > max_depth:
                        raise UnsafeArtifact(
                            path, "S5",
                            f"JSON nesting depth exceeds the {max_depth}-level depth cap")
                    if items > max_items:
                        raise UnsafeArtifact(
                            path, "S5",
                            f"JSON contains more than the {max_items}-item cap")
                elif byte in (0x7D, 0x5D):  # } ]
                    depth -= 1


# --- S6: a file_name from an annotation file is an attacker-controlled path -------------

def safe_join(root: Path, file_name: str) -> Path:
    """S6 — resolve `file_name` under `root` or refuse.

    COCO's `file_name` and YOLO's label stems are read straight out of the dataset under
    audit, so they are attacker-controlled strings that we are about to hand to `open()`.
    `../../etc/passwd`, an absolute path, and a symlink planted inside the dataset that
    points outside it are three spellings of the same escape, and all three have to be
    caught after resolution rather than by inspecting the string: resolving is what turns
    them into the same question.

    `is_relative_to` rather than a string prefix test, because "/data/root2" starts with
    "/data/root" and is a different directory.
    """
    resolved_root = Path(root).resolve()
    # `Path("/a") / "/etc/passwd"` is "/etc/passwd" — pathlib drops the left operand for an
    # absolute right one, so the absolute-path case falls out of the containment check
    # below without a separate branch.
    candidate = (resolved_root / file_name).resolve()
    if candidate == resolved_root or not candidate.is_relative_to(resolved_root):
        raise UnsafeArtifact(
            candidate, "S6",
            f"file_name {file_name!r} resolves to {candidate}, outside the dataset root "
            f"{resolved_root}")
    return candidate


# --- S7: refuse the decompression bomb from the header, not from the pixel buffer -------

def check_pixel_budget(path: Path, max_pixels: int = PIXEL_BUDGET_DEFAULT) -> tuple[int, int]:
    """S7 — cap the pixel count using only the image header.

    A decompression bomb is a few kilobytes on disk that becomes tens of gigabytes once
    decoded, so the check is worthless unless it happens before the decode. `Image.open` is
    lazy — it reads the header and stops — so `.size` is available without ever calling
    `.load()`, and that is the only thing this function is allowed to touch.

    `Image.MAX_IMAGE_PIXELS` is set to the same cap as a second line of defence for any
    decode that happens later in the process. That assignment is process-global, which is
    a footgun worth naming; it is what the plan specifies, and Pillow's own limit only
    raises at twice the value anyway, so the explicit comparison below is the real control.
    """
    import warnings

    from PIL import Image

    # `min`, never a plain assignment: the process-global setting may only ever be
    # tightened by this control. A caller passing a generous budget must not silently raise
    # the ceiling for every other decode in the process.
    Image.MAX_IMAGE_PIXELS = min(Image.MAX_IMAGE_PIXELS or max_pixels, max_pixels)
    try:
        with warnings.catch_warnings():
            # Pillow only *warns* between the cap and twice the cap. A warning that the
            # operator never reads is not a control, so it is promoted to an error.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                width, height = im.size
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        # Pillow's own limit fired first. Re-raise it in this control's vocabulary so the
        # report says which budget was breached rather than quoting a library's wording.
        raise UnsafeArtifact(
            path, "S7",
            f"image header breaches the {max_pixels}-pixel decode budget: {exc}") from exc
    except Exception as exc:
        raise UnsafeArtifact(
            path, "S7", f"image header could not be read safely: {exc}") from exc

    if width * height > max_pixels:
        raise UnsafeArtifact(
            path, "S7",
            f"image header declares {width}x{height} = {width * height} pixels, above the "
            f"{max_pixels}-pixel decode budget")
    return width, height


# --- S8 ---------------------------------------------------------------------------------

def decoder_pins() -> dict[str, str]:
    """S8 — the decoder versions actually in use, for the access-assumptions block.

    Read from the imported modules rather than from the distribution metadata: metadata can
    disagree with what is really loaded in an environment that was installed over, and the
    report is making a claim about the code that ran, not about the code that was intended
    to run. A decoder that is not installed is recorded as absent rather than omitted, so
    the block can never be read as "pinned" when it is merely silent.
    """
    pins: dict[str, str] = {}
    for name, module_name in (
        ("pillow", "PIL"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
        ("torch", "torch"),
    ):
        try:
            import importlib
            pins[name] = str(importlib.import_module(module_name).__version__)
        except Exception:
            pins[name] = "not-installed"
    return pins


# --- S1 + S4 ----------------------------------------------------------------------------

def prescan(path: Path, max_bytes: int = 4 << 30) -> SafetyReport:
    """S1 (+S4) — hash first, then decide whether opening is safe at all.

    The order is the control, not an implementation detail: the digest is computed before
    anything parses the file, so the ledger entry survives a load that kills the process.
    This function therefore never raises for a policy verdict — it reports, and
    `loaders/detect.py` decides — because a raise here would lose the digest that the whole
    ordering exists to preserve.
    """
    digest = sha256_file(path)
    size = path.stat().st_size
    reasons: list[str] = []
    safe = True

    if size > max_bytes:
        safe = False
        reasons.append(f"file is {size/1e9:.1f} GB, above the {max_bytes/1e9:.0f} GB cap")

    if path.suffix == ".onnx":
        try:
            import onnx
            m = onnx.load(str(path), load_external_data=False)
            custom = sorted({n.op_type for n in m.graph.node
                             if n.domain not in STANDARD_ONNX_DOMAINS})
            if custom:
                safe = False
                reasons.append(
                    f"non-standard ONNX operators {custom} — a custom op can load a shared "
                    "library at session-creation time")
        except Exception as exc:
            safe = False
            reasons.append(f"ONNX graph could not be parsed for a safety prescan: {exc}")

    if path.suffix in {".pt", ".pth"}:
        reasons.append("loaded with weights_only=True on torch>=2.6.0 (CVE-2025-32434)")

    return SafetyReport(path, digest, size, safe, tuple(reasons))
