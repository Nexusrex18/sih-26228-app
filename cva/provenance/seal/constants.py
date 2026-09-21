"""Wire-format constants for cva-seal/1 (Module C plan §5, decisions D1/D3/D5).

Every hash use has a distinct first byte, except `record_hash`, whose input is JSON text and so always
starts with 0x7B ('{'): it cannot collide with any input starting 0x00–0x03. That observation is why
`record_hash` needs no prefix (plan §5.5).
"""
from __future__ import annotations

VERSION = "cva-seal/1"

# --- hash domain-separation bytes ----------------------------------------------------------
DOMAIN_LEAF = b"\x00"            # RFC 6962/9162 Merkle leaf
DOMAIN_NODE = b"\x01"            # RFC 6962/9162 Merkle internal node
DOMAIN_LINK = b"\x02"            # chain link:  SHA-256(0x02 ‖ record_hash(prev) ‖ sig_bytes(prev))
DOMAIN_MANIFEST = b"\x03"        # genesis:     SHA-256(0x03 ‖ JCS(deployment_manifest))   (D5)

# --- signature domain tags (D1): sign TAG ‖ canonical_bytes, never bare canonical_bytes ------
TAG_RECORD = f"{VERSION} record\n".encode("ascii")
TAG_CHECKPOINT = f"{VERSION} checkpoint\n".encode("ascii")
TAG_COSIGN = f"{VERSION} cosign\n".encode("ascii")
TAG_ROTATION_POP = f"{VERSION} rotation-pop\n".encode("ascii")
TAG_WITNESS = f"{VERSION} witness\n".encode("ascii")

# --- canonical profile limits (D3) -----------------------------------------------------------
INT_MAX = 2**53 - 1              # JCS / IEEE-754 exact-integer range
MAX_DEPTH = 8                    # nesting of objects/arrays
MAX_RECORD_BYTES = 64 * 1024     # a record bigger than this is an attack, not an inference

TIMESTAMP_LEN = 27               # YYYY-MM-DDTHH:MM:SS.ffffffZ
