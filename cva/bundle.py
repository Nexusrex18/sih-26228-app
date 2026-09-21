"""`make bundle` -> `cva-bundle.tar`: everything the air-gapped host installs from.

Backend owns the TAR (wheelhouse, generated manifest, weights, fixtures); Network owns the
Dockerfile and image (plan C10, §3.3). The boundary is the tar.

    wheelhouse/            every wheel, built once on the networked machine (`make wheelhouse`)
    requirements.lock      the pins those wheels were resolved from
    cva-src/               tracked source + the vendored backbone (weights are gitignored)
    fixtures/              the selftest / demo corpora, produced by cva.fixtures
    MANIFEST.json          GENERATED: every external artefact, its source, version, licence
    INSTALL.md             the offline install, three commands

The size and CUDA checks run BEFORE the tar is written as well as after: a 17x overshoot
(§5.7, R5) is cheaper to discover as a refusal than as a 3 GB file.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from cva.features import vendor

ROOT = Path(__file__).resolve().parents[1]
TAR = ROOT / "cva-bundle.tar"
LIMIT_BYTES = 4 * 1000**3          # V12: the tar must be < 4 GB
FORBIDDEN = re.compile(r"(^|/)(nvidia[-_]|triton)", re.I)
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI = "https://pypi.org/simple"

INSTALL = """# Offline install

    python3 -m venv .venv && . .venv/bin/activate
    pip install --no-index --find-links=wheelhouse -r requirements.lock
    pip install --no-index --no-build-isolation --no-deps -e cva-src
    python -m cva.cli selftest        # add `unshare -rn` in front for the OS-level proof

Nothing here needs a network, and `selftest` fails if anything tries.
"""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _licence(wheel: Path) -> str:
    """The licence a wheel DECLARES in its own METADATA — a declaration, not a legal opinion."""
    try:
        with zipfile.ZipFile(wheel) as z:
            name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
            meta = z.read(name).decode("utf-8", "replace")
    except (StopIteration, zipfile.BadZipFile):
        return "unknown"
    if wheel.name.startswith("cva-"):
        return "not declared (this project's own package)"
    for key in ("License-Expression", "License"):
        m = re.search(rf"^{key}: (.+)$", meta, re.M)
        if m and m.group(1).strip() not in ("", "UNKNOWN") and len(m.group(1)) < 120:
            return m.group(1).strip()
    cls = re.findall(r"^Classifier: License :: (?:OSI Approved :: )?(.+)$", meta, re.M)
    return "; ".join(cls) if cls else "unknown"


def wheel_rows(wheelhouse: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for w in sorted(wheelhouse.glob("*.whl")):
        parts = w.name[:-4].split("-")
        version = parts[1]
        rows.append({
            "artefact": parts[0].replace("_", "-"), "kind": "wheel", "version": version,
            "file": f"wheelhouse/{w.name}", "sha256": _sha256(w), "licence": _licence(w),
            "source": CPU_INDEX if "+cpu" in version else PYPI})
    return rows


def _refuse_cuda(names: list[str], where: str) -> None:
    bad = [n for n in names if FORBIDDEN.search(n)]
    if bad:
        raise SystemExit(f"CUDA wheel in the {where}: {', '.join(bad[:5])} — see §5.7")


def _source_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                         check=True).stdout.decode().split("\0")
    return [ROOT / f for f in out if f and (ROOT / f).is_file()]


def _tree_state() -> dict[str, object]:
    """The bundle's source snapshot is `git ls-files`, so an uncommitted file is silently
    absent from it. Recording the commit and whether the tree was dirty makes that visible
    instead of surfacing as an ImportError on the air-gapped host."""
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"code_commit": git("rev-parse", "--short", "HEAD") or "unknown",
            "source_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")
                                      or git("ls-files", "--others", "--exclude-standard"))}


def _add_bytes(tar: tarfile.TarFile, arcname: str, blob: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(blob)
    info.mtime = 0                       # a bundle is reproducible bytes, not a timestamp
    tar.addfile(info, io.BytesIO(blob))


def main() -> int:
    wheelhouse, lock = ROOT / "wheelhouse", ROOT / "requirements.lock"
    if not lock.is_file() or not wheelhouse.is_dir():
        raise SystemExit("no requirements.lock / wheelhouse/ — run `make wheelhouse` first")
    _refuse_cuda(lock.read_text().splitlines(), "lockfile")

    # cva itself goes in as a wheel too, so the lock's runtime deps are the only thing the
    # install has to fetch from the wheelhouse.
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    "-q", "-w", str(wheelhouse), str(ROOT)], check=True)
    _refuse_cuda([p.name for p in wheelhouse.glob("*")], "wheelhouse")

    rows = wheel_rows(wheelhouse) + vendor.manifest()
    missing = [r["artefact"] for r in rows
               if r["kind"] != "wheel" and not (r["present"] and r["digest_matches"])]
    if missing:
        raise SystemExit(f"vendored artefact absent or digest mismatch: {', '.join(map(str, missing))}")
    tree = _tree_state()
    if tree["source_tree_dirty"]:
        print("WARNING: uncommitted or untracked files are NOT in cva-src/ — commit first.",
              file=sys.stderr)
    manifest = {"generated_by": "cva.bundle", "torch_index": CPU_INDEX, **tree,
                "artefacts": rows}

    with tempfile.TemporaryDirectory(prefix="cva-bundle-") as tmp:
        from cva.fixtures import build as build_fixtures
        fx = build_fixtures(Path(tmp) / "fixtures")
        with tarfile.open(TAR, "w") as tar:
            tar.add(wheelhouse, arcname="wheelhouse")
            tar.add(lock, arcname="requirements.lock")
            seen: set[Path] = set()
            for f in _source_files():
                tar.add(f, arcname=f"cva-src/{f.relative_to(ROOT)}")
                seen.add(f)
            tar.add(ROOT / "vendor", arcname="cva-src/vendor")   # weights are gitignored
            tar.add(fx.root, arcname="fixtures")
            _add_bytes(tar, "MANIFEST.json", json.dumps(manifest, indent=2).encode())
            _add_bytes(tar, "INSTALL.md", INSTALL.encode())

    size = TAR.stat().st_size
    with tarfile.open(TAR) as tar:
        _refuse_cuda(tar.getnames(), "tar")
    print(f"{TAR.name}: {size / 1e9:.2f} GB, {len(rows)} artefacts in MANIFEST.json")
    if size >= LIMIT_BYTES:
        raise SystemExit(f"{TAR.name} is {size / 1e9:.2f} GB; the target is < 4 GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
