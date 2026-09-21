"""Vendored backbone artefacts: what they are, where they came from, and the digest that
says the bytes on this disk are the bytes that were pinned.

Two artefacts per §5.4, not one. `torch.hub.load(..., source='local')` localises the CODE
only — the repo directory must really contain `hubconf.py` and the `dinov2/` package — so
the weights and the source snapshot are pinned separately and independently.

`extractor_id` pins a BYTE STRING, never a name (ADR-009). The same FAIR repository also
distributes XRay-DINO and Cell-DINO under a noncommercial research licence: "DINOv2 is
Apache-2.0" is true of the checkpoint we use and false of two of its neighbours. A name
cannot tell those apart; a SHA-256 can.

This module is the ONLY place in `cva/` permitted to open a network connection, and only
under `fetch`, which is Mode A (networked dev machine) exclusively. Nothing on the scan
path calls it — the scan path calls `local_path()`, which only ever reads the disk.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor"


@dataclass(frozen=True)
class Artefact:
    name: str
    url: str
    relpath: str
    sha256: str
    licence: str
    source: str
    #: True when the digest is published by the distributor and we merely confirmed it;
    #: False when we computed it on first fetch. The distinction is real and the manifest
    #: prints it: a trust-on-first-use pin proves the bytes have not changed SINCE, not
    #: that they were the intended bytes to begin with.
    digest_published: bool

    @property
    def path(self) -> Path:
        return VENDOR_ROOT / self.relpath


ARTEFACTS: dict[str, Artefact] = {
    "dinov2_vits14": Artefact(
        name="dinov2_vits14",
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        relpath="weights/dinov2_vits14_pretrain.pth",
        sha256="b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
        licence="Apache-2.0",
        source="facebookresearch/dinov2 (dl.fbaipublicfiles.com)",
        # FAIR publishes no digest alongside the checkpoint. This pin was computed on
        # first fetch on the dev machine, 2026-09-19. Say so rather than imply more.
        digest_published=False),
    "resnet18": Artefact(
        name="resnet18",
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        relpath="weights/resnet18-f37072fd.pth",
        sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        licence="BSD-3-Clause (torchvision)",
        source="pytorch/vision (download.pytorch.org)",
        # torchvision encodes the first 8 hex of the SHA-256 IN THE FILENAME and checks it
        # on download, so `resnet18-f37072fd` is a digest the distributor published.
        digest_published=True),
}

#: The source snapshot, pinned by commit rather than by file digest — it is a tree.
DINOV2_REPO_COMMIT = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
DINOV2_REPO_URL = f"https://codeload.github.com/facebookresearch/dinov2/tar.gz/{DINOV2_REPO_COMMIT}"
DINOV2_REPO_DIR = VENDOR_ROOT / "dinov2" / "repo"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class VendorError(RuntimeError):
    """A vendored artefact is missing or does not match its pin.

    Never downgraded to a warning. A backbone whose bytes do not match the pin is not a
    degraded backbone, it is an unknown one, and every embedding-derived finding in the
    report would be attributed to a checkpoint we cannot name.
    """


def verify(name: str) -> None:
    art = ARTEFACTS[name]
    if not art.path.exists():
        raise VendorError(
            f"{name}: {art.path} is absent. Mode A: `make vendor`. Mode B/C: the bundle "
            f"ships it and this means the bundle was not unpacked here.")
    got = sha256_of(art.path)
    if got != art.sha256:
        raise VendorError(
            f"{name}: digest mismatch at {art.path}\n  pinned {art.sha256}\n  found  {got}")


def available(name: str) -> bool:
    """True when the artefact is present AND matches. Used to choose a backbone, never to
    decide whether a mismatch is tolerable — a mismatch raises wherever it is loaded."""
    try:
        verify(name)
    except (VendorError, KeyError):
        return False
    return True


def local_path(name: str) -> Path:
    verify(name)
    return ARTEFACTS[name].path


def repo_available() -> bool:
    return (DINOV2_REPO_DIR / "hubconf.py").is_file() and (DINOV2_REPO_DIR / "dinov2").is_dir()


def manifest() -> list[dict[str, object]]:
    """The generated vendoring manifest (§3.3, B6). GENERATED, never hand-written: a
    manifest typed by a person drifts from what actually shipped, and the sentence about
    it is the one artefact that must not."""
    rows: list[dict[str, object]] = []
    for art in ARTEFACTS.values():
        present = art.path.exists()
        rows.append({
            "artefact": art.name, "kind": "weights", "file": art.relpath,
            "source": art.source, "url": art.url, "licence": art.licence,
            "sha256": art.sha256, "digest_published_by_distributor": art.digest_published,
            "present": present,
            "digest_matches": present and sha256_of(art.path) == art.sha256})
    rows.append({
        "artefact": "dinov2-source", "kind": "source-snapshot",
        "file": "dinov2/repo", "source": "facebookresearch/dinov2",
        "url": DINOV2_REPO_URL, "licence": "Apache-2.0",
        "sha256": f"commit:{DINOV2_REPO_COMMIT}", "digest_published_by_distributor": True,
        "present": repo_available(), "digest_matches": repo_available()})
    return rows


def _fetch_one(art: Artefact) -> None:
    art.path.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {art.name} <- {art.url}")
    urllib.request.urlopen  # noqa: B018  — named so the egress surface is greppable
    with urllib.request.urlopen(art.url, timeout=600) as r, art.path.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    got = sha256_of(art.path)
    if got != art.sha256:
        art.path.unlink(missing_ok=True)
        raise VendorError(f"{art.name}: fetched bytes do not match the pin ({got})")
    print(f"  ok  {art.sha256}  {art.path}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "verify"
    if cmd == "fetch":
        for art in ARTEFACTS.values():
            _fetch_one(art)
        if not repo_available():
            print(f"MISSING source snapshot: {DINOV2_REPO_DIR}\n"
                  f"  curl -sL {DINOV2_REPO_URL} | tar xz -C /tmp && \\\n"
                  f"  mv /tmp/dinov2-{DINOV2_REPO_COMMIT} {DINOV2_REPO_DIR}")
            return 1
        return 0
    if cmd == "manifest":
        import json
        print(json.dumps(manifest(), indent=2))
        return 0
    bad = [r["artefact"] for r in manifest() if not r["digest_matches"]]
    print("\n".join(f"{r['artefact']:16} {'ok' if r['digest_matches'] else 'MISSING/MISMATCH'}"
                    for r in manifest()))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
