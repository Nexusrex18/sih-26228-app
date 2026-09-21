"""The image's files agree with each other and with the code they configure (§7.9).

These run without docker. They catch the drift that only shows up after a 20-minute build:
a uid in the policy that the Dockerfile never creates, a role the entrypoint documents and
does not handle, a base image that quietly floats on a tag.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"


def test_the_entrypoint_is_valid_posix_sh():
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("no sh on this machine")
    proc = subprocess.run([sh, "-n", str(DOCKER / "entrypoint.sh")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_the_policy_loads_and_grants_only_what_each_role_needs():
    from cva.ledgerd.policy import SCAN_RECORD, ANALYST_EVENT, from_mapping

    pol = from_mapping(json.loads((DOCKER / "ledgerd-policy.json").read_text()))
    assert pol.writers == {10002: frozenset({ANALYST_EVENT}),
                           10003: frozenset({SCAN_RECORD})}
    assert pol.readers == frozenset({10002})
    assert pol.same_uid_ok is False
    assert 10001 not in pol.writers, "the key holder signs; it never originates a record"


def test_every_uid_the_policy_names_is_created_by_the_dockerfile():
    dockerfile = (DOCKER / "Dockerfile").read_text()
    created = {int(u) for u in re.findall(r"useradd[^\n]*--uid (\d+)", dockerfile)}
    policy = json.loads((DOCKER / "ledgerd-policy.json").read_text())
    named = {int(u) for u in policy["writers"]} | set(policy["readers"])
    assert named <= created, named - created


def test_the_entrypoint_checks_the_same_uids_the_policy_names():
    ep = (DOCKER / "entrypoint.sh").read_text()
    assert re.search(r"web\)\s*\n\s*need_uid 10002", ep)
    assert re.search(r"scan\)\s*\n\s*need_uid 10003", ep)
    assert re.search(r"ledgerd\)\s*\n\s*need_uid 10001", ep)


def test_every_documented_role_is_handled():
    ep = (DOCKER / "entrypoint.sh").read_text()
    documented = set(re.findall(r"^#   (\w+)\s", ep, re.M))
    handled = set(re.findall(r"^  (\w+)\)$", ep, re.M))
    assert documented and documented <= handled, documented - handled


def test_the_base_image_is_never_a_bare_tag():
    """The Dockerfile's default must fail to resolve, and image.mk must pass a digest."""
    dockerfile = (DOCKER / "Dockerfile").read_text()
    default = re.search(r"^ARG BASE_IMAGE=(\S+)", dockerfile, re.M).group(1)
    assert "@sha256:" not in default and ":" not in default.split("/")[-1], (
        "a usable default here would let a hand-run build float on a tag")
    mk = (DOCKER / "image.mk").read_text()
    assert "BASE_IMAGE=$(BASE_REPO)@$$(cat $(DIGEST_FILE))" in mk


def test_the_web_role_publishes_to_host_loopback_in_every_documented_command():
    """S9 rests on the publish spec once the entrypoint binds 0.0.0.0 in a container."""
    for doc in (DOCKER / "hardening.md", ROOT / "docs" / "SETUP.md"):
        if not doc.exists():
            continue
        for line in doc.read_text().splitlines():
            if re.search(r"(^|\s)-p\s", line):
                assert re.search(r"-p 127\.0\.0\.1:", line), f"{doc.name}: {line.strip()}"


def test_the_image_bakes_no_key():
    dockerfile = (DOCKER / "Dockerfile").read_text()
    for line in dockerfile.splitlines():
        if line.lstrip().startswith(("COPY", "ADD")):
            assert "key" not in line.lower(), line
