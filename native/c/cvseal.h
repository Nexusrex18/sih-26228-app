/* cvseal — the C core of the inference provenance seal (Module C, gate C9).
 *
 * Wire format: cva-seal/1 (the published spec: Plan/cva-seal-spec-v1.md in the notes repo, https://github.com/Suvrat1629/sih26228-notes). This library is a second, native
 * implementation of the WRITER side (canonicalise, sign, chain, Merkle, checkpoint, rotate, append to the same
 * SQLite store the Python reference uses) plus a compact verifier of the chain. Deliberate narrowing, stated:
 * it enforces the JSON profile, the header, the cryptography and the analyst_event rules (an override without a
 * justification is refused); the other per-section schemas (model, input, output ...) are validated by the Python
 * verifier, which remains the authority on "is this a well-formed record".
 *
 * Dependencies: libsodium (ISC) for SHA-256 and pure Ed25519, sqlite3. No other.
 * Every function returns 0 on success, non-zero on failure; details in cvs_errmsg(). Not thread-safe per handle.
 */
#ifndef CVSEAL_H
#define CVSEAL_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CVS_VERSION "cva-seal/1"
#define CVS_OK 0
#define CVS_ERR 1

typedef struct cvs_ledger cvs_ledger;

/* Global init (libsodium). Idempotent. */
int cvs_init(void);
const char *cvs_errmsg(void);

/* ---- canonical JSON (spec section 2) ------------------------------------------------------------------- */
/* Parse `in` (any JSON text: whitespace and key order are free; escapes are decoded) into the cva-seal/1
 * profile and write its canonical bytes to a malloc'd buffer. Refuses floats, non-ASCII, bad keys, duplicate
 * keys, mixed arrays, null in arrays, depth > 8, |n| > 2^53-1. Caller frees *out with cvs_free. */
int cvs_canonicalise(const char *in, size_t n, char **out, size_t *out_n);
/* Like cvs_canonicalise, but the input must ALREADY be the canonical bytes (a stored record). */
int cvs_parse_canonical(const char *in, size_t n);
void cvs_free(void *p);

/* ---- primitives ------------------------------------------------------------------------------------------ */
void cvs_sha256(const uint8_t *data, size_t n, uint8_t out[32]);
/* key_id = SHA-256(public key), 64 lowercase hex + NUL */
void cvs_key_id(const uint8_t pub[32], char out[65]);
void cvs_keypair_from_seed(const uint8_t seed[32], uint8_t pub[32], uint8_t sk[64]);

/* ---- one record ---------------------------------------------------------------------------------------------- */
/* Build, sign and serialise a record. `body_json` is an object holding exactly the type's sections.
 * `prev_hash` is the 64-hex prev_record_hash (a link, or the manifest hash for genesis). Writes the stored
 * bytes (malloc'd, no trailing newline). record_hash and link are optional outputs. */
int cvs_seal_record(const char *type, uint64_t seq, const char *prev_hash, const uint8_t pub[32],
                    const uint8_t sk[64], const char *created_at_utc, const uint8_t nonce[16],
                    const char *body_json, char **stored, size_t *stored_n, uint8_t record_hash[32],
                    char link_hex[65]);
/* Merkle leaf hash of stored record bytes. */
void cvs_leaf_hash(const uint8_t *stored, size_t n, uint8_t out[32]);
/* RFC 6962 root over `n` leaf hashes. */
void cvs_merkle_root(const uint8_t *leaves, size_t n, uint8_t out[32]);

/* ---- the ledger (same SQLite store as the Python reference: cva-seal-store/2) -------------------------------- */
int cvs_ledger_init(const char *path, const uint8_t seed[32], const char *manifest_json, cvs_ledger **out);
int cvs_ledger_open(const char *path, const uint8_t seed[32], cvs_ledger **out);
/* as cvs_ledger_init, with the genesis timestamp and nonce fixed (reproducible fixtures) */
int cvs_ledger_init_at(const char *path, const uint8_t seed[32], const char *manifest_json, const char *now,
                       const uint8_t *nonce, cvs_ledger **out);
void cvs_ledger_close(cvs_ledger *l);
/* `now`/`nonce` may be NULL (real clock / CSPRNG); tests pass fixed values for reproducibility. A checkpoint is
 * appended automatically on the manifest's cadence. Returns the seq written in *seq_out (may be NULL). */
int cvs_ledger_append(cvs_ledger *l, const char *type, const char *body_json, const char *now,
                      const uint8_t *nonce, uint64_t *seq_out);
/* Store a payload (content-addressed) in the same store; writes its 64-hex address. */
int cvs_ledger_put_payload(cvs_ledger *l, const uint8_t *data, size_t n, char address[65]);
/* Rotate to a new key: key_rotation signed by the outgoing key with the incoming key's proof of possession. */
int cvs_ledger_rotate(cvs_ledger *l, const uint8_t new_seed[32], const char *now, const uint8_t *nonce);
uint64_t cvs_ledger_size(cvs_ledger *l);

/* ---- verification of the chain -------------------------------------------------------------------------------- */
/* Verify a ledger file (SQLite store or strict JSONL export): canonical form, seq, active key (with rotations
 * and proofs of possession), signature, links, genesis commitment and checkpoint roots. `genesis_pubkey_hex` is
 * the trust root's ledger key (64 hex) — a record names its signer only by hash. Returns 0 if intact; otherwise
 * non-zero with the first failure in cvs_errmsg(), naming the record. If `expect_manifest_hash_hex` is non-NULL
 * the genesis must commit to it. */
int cvs_verify_file(const char *path, const char *genesis_pubkey_hex, const char *expect_manifest_hash_hex,
                    uint64_t *records_out);

#ifdef __cplusplus
}
#endif
#endif
