"""Regenerate the vendored test vectors from their primary sources.

Vectors are PARSED out of the source text, never typed by hand (Module C plan §11.1: "copy them
from the RFC — do not type them from memory"). Each output file records its source URL and the
SHA-256 of the exact bytes that were parsed, so a reviewer can re-fetch and compare.

Dev-only tool: uses the network and is never imported by the product (ADR-005). Run from the repo
root:  .venv/bin/python spec/vectors/_extract.py
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CT_COMMIT = "a490ef305a5bc3e556495fd824681d090d832d27"     # transparency-dev/merkle, pinned
SOURCES = {
    "rfc8032": "https://www.rfc-editor.org/rfc/rfc8032.txt",
    "rfc8785": "https://www.rfc-editor.org/rfc/rfc8785.txt",
    "ct_constants": f"https://raw.githubusercontent.com/transparency-dev/merkle/{CT_COMMIT}/testonly/constants.go",
    "ct_license": f"https://raw.githubusercontent.com/transparency-dev/merkle/{CT_COMMIT}/LICENSE",
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def src_meta(name: str, data: bytes) -> dict:
    return {"url": SOURCES[name], "sha256": hashlib.sha256(data).hexdigest()}


def hexblock(lines: list[str]) -> str:
    return "".join(l.strip() for l in lines)


def rfc8032(text: str, meta: dict) -> dict:
    """RFC 8032 §7.1 — the pure Ed25519 vectors (TEST 1, 2, 3, 1024). Ed25519ph (`SHA(abc)`) is a
    different algorithm and is deliberately excluded: the seal uses PureEdDSA only."""
    # Strip RFC page furniture (form feed, running header, page footer): a long hex block can span a
    # page break, and without this the 1023-byte message silently truncates.
    text = "\n".join(l for l in text.splitlines()
                     if "\x0c" not in l and not re.search(r"\[Page \d+\]$", l)
                     and not re.match(r"^RFC 8032 .*January 2017$", l))
    sec = text.split("7.1.  Test Vectors for Ed25519")[2]           # [0]=TOC line, [1]=section head
    sec = sec.split("7.2.  Test Vectors for Ed25519ctx")[0]
    vectors = []
    for name, body in re.findall(r"-----TEST (\w+)\n(.*?)(?=\n   -----TEST |\Z)", sec, re.DOTALL):
        if name == "SHA(abc)":
            continue
        def field(label: str, body: str = body) -> str:
            m = re.search(rf"{label}[^\n]*:\n((?:(?:   [0-9a-f]+)?\n)*)", body)
            return hexblock(m.group(1).splitlines()) if m else ""
        vectors.append({"name": f"TEST {name}", "secret_key": field("SECRET KEY"),
                        "public_key": field("PUBLIC KEY"), "message": field("MESSAGE"),
                        "signature": field("SIGNATURE")})
    return {"source": meta, "section": "RFC 8032 §7.1", "vectors": vectors}


def rfc8785_numbers(text: str, meta: dict) -> dict:
    """RFC 8785 Appendix B, Table 1."""
    app = text.split("Appendix B.  Number Serialization Samples")[2].split("Appendix C.")[0]
    rows = []
    for ieee, js, comment in re.findall(r"^   \| ([0-9a-f]{16}) \| (.*?) \|\s*(.*?)\s*\|$", app, re.MULTILINE):
        js = js.strip()
        rows.append({"ieee754": ieee, "expected": js or None, "comment": comment.strip(),
                     "must_error": js == ""})            # NaN / Infinity rows carry no JSON form
    return {"source": meta, "section": "RFC 8785 Appendix B", "rows": rows}


def rfc8785_sorting(text: str, meta: dict) -> dict:
    """§3.2.2 input object, and its canonical bytes as printed in §3.2.4."""
    body32 = text.split("3.2.4.  UTF-8 Generation")[2].split("4.  IANA")[0]
    # Bound the parse to the hex block itself: the next sentence ("...intended to be usable...")
    # contains the word "be", which is a valid hex byte.
    block = body32.split("hexadecimal notation:")[1].split("This data is intended")[0]
    hexbytes = "".join(re.findall(r"\b[0-9a-f]{2}\b", block))
    sec = text.split("3.2.2.  Serialization of Primitive Data Types")[2].split("If the parsed data")[0]
    obj = sec.split("parsed:")[1].strip()
    return {"source": meta, "section": "RFC 8785 §3.2.2 / §3.2.3 / §3.2.4",
            "input_json": obj, "expected_utf8_hex": hexbytes}


def ct_merkle(constants: bytes, licence: bytes, meta: dict, lic_meta: dict) -> dict:
    src = constants.decode()
    def block(fn: str) -> str:
        return src.split(f"func {fn}()")[1].split("\n}\n")[0]
    leaves = re.findall(r'hd\("([0-9a-f]*)"\)', block("LeafInputs"))
    empty = re.findall(r'hd\("([0-9a-f]+)"\)', block("EmptyRootHash"))[0]
    # RootHashes() lists EmptyRootHash() first as a call, not an hd("…") literal.
    roots = [empty] + re.findall(r'hd\("([0-9a-f]+)"\)', block("RootHashes"))
    nodes_src = block("NodeHashes")
    levels = [re.findall(r'hd\("([0-9a-f]+)"\)', lvl) for lvl in nodes_src.split("}, {")]
    return {"source": meta, "licence": {**lic_meta, "spdx": "Apache-2.0",
            "notice": "Copyright 2019 Google LLC. All Rights Reserved. Licensed under the Apache "
                      "License, Version 2.0. Values below are copied from testonly/constants.go."},
            "note": "Inclusion/consistency proof vectors in this repo are expressed as node "
                    "coordinates, not hash bytes, so they are NOT vendored here. Proofs are covered "
                    "by the differential + exhaustive tests (plan §11.2).",
            "leaf_inputs": leaves, "empty_root": empty,
            "root_hashes_by_size": roots,             # index = tree size, starting at the empty tree
            "node_hashes_by_level": levels}


def main() -> None:
    raw = {k: fetch(u) for k, u in SOURCES.items()}
    t8032, t8785 = raw["rfc8032"].decode(), raw["rfc8785"].decode()
    out = {
        HERE / "rfc8032.json": rfc8032(t8032, src_meta("rfc8032", raw["rfc8032"])),
        HERE / "rfc8785" / "numbers.json": rfc8785_numbers(t8785, src_meta("rfc8785", raw["rfc8785"])),
        HERE / "rfc8785" / "sorting.json": rfc8785_sorting(t8785, src_meta("rfc8785", raw["rfc8785"])),
        HERE / "merkle_ct.json": ct_merkle(raw["ct_constants"], raw["ct_license"],
                                           src_meta("ct_constants", raw["ct_constants"]),
                                           src_meta("ct_license", raw["ct_license"])),
    }
    for path, obj in out.items():
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
        print("wrote", path.relative_to(HERE.parent.parent))


if __name__ == "__main__":
    main()
