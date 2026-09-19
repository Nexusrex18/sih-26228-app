"""Content-addressed payload store (plan §7.7)."""
from __future__ import annotations

import hashlib
import os

import pytest

from cva.provenance.seal.errors import PayloadCorrupt, PayloadMissing
from cva.provenance.seal.payloads import PayloadStore, hex_of, ref_for


def test_a_payload_is_addressed_by_the_sha256_of_its_bytes_and_laid_out_by_prefix(tmp_path):
    ps = PayloadStore(tmp_path / "p")
    data = b'{"task":"classify"}'
    ref = ps.put(data)
    h = hashlib.sha256(data).hexdigest()
    assert ref == f"sha256:{h}" == ref_for(data)
    assert ps.path_for(ref) == tmp_path / "p" / h[:2] / f"{h[2:]}.json"
    assert ps.path_for(ref).read_bytes() == data and ps.has(ref) and ps.get(ref) == data


def test_put_is_idempotent_and_never_rewrites(tmp_path):
    ps = PayloadStore(tmp_path)
    ref = ps.put(b"same")
    mtime = ps.path_for(ref).stat().st_mtime_ns
    assert ps.put(b"same") == ref
    assert ps.path_for(ref).stat().st_mtime_ns == mtime


def test_get_reverifies_the_address_and_reports_corruption(tmp_path):
    ps = PayloadStore(tmp_path)
    ref = ps.put(b"honest payload")
    ps.path_for(ref).write_bytes(b"edited payload")            # someone edits the stored file (T1b)
    with pytest.raises(PayloadCorrupt, match="hash to"):
        ps.get(ref)


def test_a_corrupt_existing_file_is_reported_not_overwritten(tmp_path):
    ps = PayloadStore(tmp_path)
    ref = ps.put(b"honest payload")
    ps.path_for(ref).write_bytes(b"edited payload")
    with pytest.raises(PayloadCorrupt, match="does not match"):
        ps.put(b"honest payload")
    assert ps.path_for(ref).read_bytes() == b"edited payload"   # evidence left untouched


def test_a_missing_payload_is_reported_as_missing_never_as_empty(tmp_path):
    with pytest.raises(PayloadMissing):
        PayloadStore(tmp_path).get("sha256:" + "0" * 64)


@pytest.mark.parametrize("bad", ["", "sha256:", "sha256:" + "A" * 64, "sha256:" + "a" * 63, "sha1:" + "a" * 64,
                                 "a" * 64, "sha256:../../etc/passwd"])
def test_references_must_be_exact_lowercase_sha256(bad):
    with pytest.raises(ValueError, match="not a payload reference"):
        hex_of(bad)


def test_a_reference_can_never_escape_the_store_root(tmp_path):
    ps = PayloadStore(tmp_path / "p")
    with pytest.raises(ValueError):
        ps.path_for("sha256:../../../../etc/passwd")


def test_deferred_sync_writes_now_and_fsyncs_later(tmp_path):
    ps = PayloadStore(tmp_path)
    refs = [ps.put(f"payload {i}".encode(), sync=False) for i in range(5)]
    assert all(ps.has(r) for r in refs)                           # readable immediately
    assert ps.pending > 0
    n = ps.sync_pending()
    assert n > 0 and ps.pending == 0
    assert ps.sync_pending() == 0                                 # nothing left


def test_synchronous_put_leaves_nothing_pending(tmp_path):
    ps = PayloadStore(tmp_path)
    ps.put(b"x", sync=True)
    assert ps.pending == 0


def test_files_are_written_exclusively(tmp_path):
    ps = PayloadStore(tmp_path)
    ref = ps.put(b"once")
    fd = None
    with pytest.raises(FileExistsError):
        fd = os.open(ps.path_for(ref), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    assert fd is None
