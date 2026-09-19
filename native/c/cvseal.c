/* cvseal — see cvseal.h. Written against the cva-seal/1 specification, not against the Python source. */
#define _POSIX_C_SOURCE 200809L
#include "cvseal.h"

#include <sodium.h>
#include <sqlite3.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INT_MAX_53 9007199254740991LL
#define MAX_DEPTH 8
#define MAX_RECORD 65536
static const char TAG_RECORD[] = "cva-seal/1 record\n";
static const char TAG_POP[] = "cva-seal/1 rotation-pop\n";

static char g_err[512];
static int fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
    return CVS_ERR;
}
const char *cvs_errmsg(void) { return g_err; }
int cvs_init(void) { return sodium_init() < 0 ? fail("libsodium failed to initialise") : CVS_OK; }
void cvs_free(void *p) { free(p); }

/* ================================ growable buffer ================================ */
typedef struct { char *p; size_t n, cap; } buf;
static int b_put(buf *b, const void *d, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t c = b->cap ? b->cap : 256;
        while (c < b->n + n + 1) c *= 2;
        char *q = realloc(b->p, c);
        if (!q) return fail("out of memory");
        b->p = q; b->cap = c;
    }
    memcpy(b->p + b->n, d, n);
    b->n += n;
    b->p[b->n] = 0;
    return CVS_OK;
}
static int b_c(buf *b, char c) { return b_put(b, &c, 1); }

/* ================================ JSON value ================================ */
typedef enum { J_NULL, J_BOOL, J_INT, J_STR, J_ARR, J_OBJ } jtype;
typedef struct jv {
    jtype t;
    int64_t i;
    char *s;
    struct jv **v;   /* array items / object values */
    char **k;        /* object keys */
    size_t n, cap;
} jv;

static jv *jnew(jtype t) { jv *x = calloc(1, sizeof *x); if (x) x->t = t; return x; }
static void jfree(jv *x) {
    if (!x) return;
    for (size_t i = 0; i < x->n; i++) { jfree(x->v[i]); if (x->k) free(x->k[i]); }
    free(x->v); free(x->k); free(x->s); free(x);
}
static int jpush(jv *o, const char *key, jv *val) {
    if (o->n == o->cap) {
        size_t c = o->cap ? o->cap * 2 : 4;
        jv **nv = realloc(o->v, c * sizeof *nv);
        if (!nv) return fail("out of memory");
        o->v = nv;
        if (o->t == J_OBJ) {
            char **nk = realloc(o->k, c * sizeof *nk);
            if (!nk) return fail("out of memory");
            o->k = nk;
        }
        o->cap = c;
    }
    if (o->t == J_OBJ) { o->k[o->n] = strdup(key); if (!o->k[o->n]) return fail("out of memory"); }
    o->v[o->n++] = val;
    return CVS_OK;
}
static jv *jstr(const char *s) { jv *x = jnew(J_STR); if (x) x->s = strdup(s); return x; }
static jv *jint(int64_t i) { jv *x = jnew(J_INT); if (x) x->i = i; return x; }
static jv *jget(const jv *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (size_t i = 0; i < o->n; i++) if (!strcmp(o->k[i], key)) return o->v[i];
    return NULL;
}
static const char *jgets(const jv *o, const char *key) { jv *x = jget(o, key); return x && x->t == J_STR ? x->s : NULL; }

/* ---- parser (profile of spec section 2.1; lenient about whitespace, key order and escapes) ---- */
typedef struct { const char *p, *e; } cur;
static void ws(cur *c) { while (c->p < c->e && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) c->p++; }

static int key_ok(const char *k) {
    if (!*k) return 0;
    for (; *k; k++) if (!((*k >= 'a' && *k <= 'z') || (*k >= '0' && *k <= '9') || *k == '_')) return 0;
    return 1;
}
static int parse_string(cur *c, char **out) {
    buf b = {0};
    c->p++; /* opening quote */
    for (;;) {
        if (c->p >= c->e) { free(b.p); return fail("unterminated string"); }
        unsigned char ch = (unsigned char)*c->p++;
        if (ch == '"') break;
        if (ch == '\\') {
            if (c->p >= c->e) { free(b.p); return fail("bad escape"); }
            char e = *c->p++;
            if (e == '"' || e == '\\' || e == '/') ch = (unsigned char)e;
            else if (e == 'u') {
                if (c->e - c->p < 4) { free(b.p); return fail("bad \\u escape"); }
                unsigned v = 0;
                for (int i = 0; i < 4; i++) {
                    char h = *c->p++;
                    v <<= 4;
                    if (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
                    else { free(b.p); return fail("bad \\u escape"); }
                }
                if (v < 0x20 || v > 0x7e) { free(b.p); return fail("string outside printable ASCII"); }
                ch = (unsigned char)v;
            } else { free(b.p); return fail("escape \\%c is outside the profile", e); }
        } else if (ch < 0x20 || ch > 0x7e) { free(b.p); return fail("string outside printable ASCII"); }
        if (b_c(&b, (char)ch)) { free(b.p); return CVS_ERR; }
    }
    if (!b.p) { b.p = strdup(""); if (!b.p) return fail("out of memory"); }
    *out = b.p;
    return CVS_OK;
}
static int parse_value(cur *c, int depth, jv **out);
static int parse_number(cur *c, jv **out) {
    int neg = 0;
    if (*c->p == '-') { neg = 1; c->p++; }
    if (c->p >= c->e || *c->p < '0' || *c->p > '9') return fail("bad number");
    if (*c->p == '0' && c->p + 1 < c->e && c->p[1] >= '0' && c->p[1] <= '9') return fail("leading zero");
    int64_t v = 0;
    int digits = 0;
    while (c->p < c->e && *c->p >= '0' && *c->p <= '9') {
        if (++digits > 16) return fail("integer outside +/-(2^53-1)");
        v = v * 10 + (*c->p++ - '0');
    }
    if (c->p < c->e && (*c->p == '.' || *c->p == 'e' || *c->p == 'E')) return fail("float — quantise to an integer first");
    if (v > INT_MAX_53) return fail("integer outside +/-(2^53-1)");
    *out = jint(neg ? -v : v);
    return *out ? CVS_OK : fail("out of memory");
}
static int kind_of(const jv *x) { return x->t; }
static int parse_value(cur *c, int depth, jv **out) {
    ws(c);
    if (c->p >= c->e) return fail("unexpected end of input");
    char ch = *c->p;
    if (ch == '{') {
        if (depth + 1 > MAX_DEPTH) return fail("nesting deeper than %d", MAX_DEPTH);
        jv *o = jnew(J_OBJ);
        if (!o) return fail("out of memory");
        c->p++;
        ws(c);
        if (c->p < c->e && *c->p == '}') { c->p++; *out = o; return CVS_OK; }
        for (;;) {
            ws(c);
            if (c->p >= c->e || *c->p != '"') { jfree(o); return fail("expected an object key"); }
            char *k = NULL;
            if (parse_string(c, &k)) { jfree(o); return CVS_ERR; }
            if (!key_ok(k)) { int r = fail("key \"%s\" is not [a-z0-9_]+", k); free(k); jfree(o); return r; }
            if (jget(o, k)) { int r = fail("duplicate key \"%s\"", k); free(k); jfree(o); return r; }
            ws(c);
            if (c->p >= c->e || *c->p != ':') { free(k); jfree(o); return fail("expected ':'"); }
            c->p++;
            jv *v = NULL;
            if (parse_value(c, depth + 1, &v)) { free(k); jfree(o); return CVS_ERR; }
            int r = jpush(o, k, v);
            free(k);
            if (r) { jfree(v); jfree(o); return CVS_ERR; }
            ws(c);
            if (c->p < c->e && *c->p == ',') { c->p++; continue; }
            if (c->p < c->e && *c->p == '}') { c->p++; break; }
            jfree(o);
            return fail("expected ',' or '}'");
        }
        *out = o;
        return CVS_OK;
    }
    if (ch == '[') {
        if (depth + 1 > MAX_DEPTH) return fail("nesting deeper than %d", MAX_DEPTH);
        jv *a = jnew(J_ARR);
        if (!a) return fail("out of memory");
        c->p++;
        ws(c);
        if (c->p < c->e && *c->p == ']') { c->p++; *out = a; return CVS_OK; }
        for (;;) {
            jv *v = NULL;
            if (parse_value(c, depth + 1, &v)) { jfree(a); return CVS_ERR; }
            if (v->t == J_NULL) { jfree(v); jfree(a); return fail("null inside an array"); }
            if (a->n && kind_of(a->v[0]) != kind_of(v)) { jfree(v); jfree(a); return fail("array of mixed types"); }
            if (jpush(a, NULL, v)) { jfree(v); jfree(a); return CVS_ERR; }
            ws(c);
            if (c->p < c->e && *c->p == ',') { c->p++; continue; }
            if (c->p < c->e && *c->p == ']') { c->p++; break; }
            jfree(a);
            return fail("expected ',' or ']'");
        }
        *out = a;
        return CVS_OK;
    }
    if (ch == '"') {
        char *s = NULL;
        if (parse_string(c, &s)) return CVS_ERR;
        jv *x = jnew(J_STR);
        if (!x) { free(s); return fail("out of memory"); }
        x->s = s;
        *out = x;
        return CVS_OK;
    }
    if (ch == '-' || (ch >= '0' && ch <= '9')) return parse_number(c, out);
    if (c->e - c->p >= 4 && !strncmp(c->p, "true", 4)) { c->p += 4; *out = jnew(J_BOOL); if (*out) (*out)->i = 1; return *out ? CVS_OK : fail("out of memory"); }
    if (c->e - c->p >= 5 && !strncmp(c->p, "false", 5)) { c->p += 5; *out = jnew(J_BOOL); return *out ? CVS_OK : fail("out of memory"); }
    if (c->e - c->p >= 4 && !strncmp(c->p, "null", 4)) { c->p += 4; *out = jnew(J_NULL); return *out ? CVS_OK : fail("out of memory"); }
    return fail("unexpected character '%c'", ch);
}
static int parse_doc(const char *in, size_t n, jv **out) {
    for (size_t i = 0; i < n; i++) if ((unsigned char)in[i] >= 0x80) return fail("non-ASCII byte at offset %zu", i);
    cur c = {in, in + n};
    jv *v = NULL;
    ws(&c);
    if (c.p >= c.e || *c.p != '{') return fail("top level must be an object");
    if (parse_value(&c, 0, &v)) return CVS_ERR;
    ws(&c);
    if (c.p != c.e) { jfree(v); return fail("trailing content after the object"); }
    *out = v;
    return CVS_OK;
}

/* ---- canonical serialiser ---- */
static int ser(const jv *x, buf *b) {
    switch (x->t) {
    case J_NULL: return b_put(b, "null", 4);
    case J_BOOL: return x->i ? b_put(b, "true", 4) : b_put(b, "false", 5);
    case J_INT: { char t[32]; int n = snprintf(t, sizeof t, "%lld", (long long)x->i); return b_put(b, t, (size_t)n); }
    case J_STR:
        if (b_c(b, '"')) return CVS_ERR;
        for (const char *p = x->s; *p; p++) {
            if ((*p == '"' || *p == '\\') && b_c(b, '\\')) return CVS_ERR;
            if (b_c(b, *p)) return CVS_ERR;
        }
        return b_c(b, '"');
    case J_ARR:
        if (b_c(b, '[')) return CVS_ERR;
        for (size_t i = 0; i < x->n; i++) { if (i && b_c(b, ',')) return CVS_ERR; if (ser(x->v[i], b)) return CVS_ERR; }
        return b_c(b, ']');
    case J_OBJ: {
        size_t *idx = malloc((x->n ? x->n : 1) * sizeof *idx);
        if (!idx) return fail("out of memory");
        for (size_t i = 0; i < x->n; i++) idx[i] = i;
        /* insertion sort by key bytes: objects here are small */
        for (size_t i = 1; i < x->n; i++) {
            size_t t = idx[i], j = i;
            while (j > 0 && strcmp(x->k[idx[j - 1]], x->k[t]) > 0) { idx[j] = idx[j - 1]; j--; }
            idx[j] = t;
        }
        int r = b_c(b, '{');
        for (size_t i = 0; !r && i < x->n; i++) {
            if (i) r = b_c(b, ',');
            if (!r) { jv key = {.t = J_STR, .s = x->k[idx[i]]}; r = ser(&key, b); }
            if (!r) r = b_c(b, ':');
            if (!r) r = ser(x->v[idx[i]], b);
        }
        free(idx);
        return r ? CVS_ERR : b_c(b, '}');
    }
    }
    return fail("unreachable");
}
static int canon_of(const jv *x, char **out, size_t *n) {
    buf b = {0};
    if (ser(x, &b)) { free(b.p); return CVS_ERR; }
    if (!b.p) { b.p = strdup(""); }
    *out = b.p; *n = b.n;
    return CVS_OK;
}
/* canonical bytes of `o` without its "signature" member */
static int canon_unsigned(const jv *o, char **out, size_t *n) {
    jv tmp = *o;
    tmp.v = malloc((o->n ? o->n : 1) * sizeof *tmp.v);
    tmp.k = malloc((o->n ? o->n : 1) * sizeof *tmp.k);
    if (!tmp.v || !tmp.k) { free(tmp.v); free(tmp.k); return fail("out of memory"); }
    size_t m = 0;
    for (size_t i = 0; i < o->n; i++) if (strcmp(o->k[i], "signature")) { tmp.v[m] = o->v[i]; tmp.k[m] = o->k[i]; m++; }
    tmp.n = m;
    int r = canon_of(&tmp, out, n);
    free(tmp.v); free(tmp.k);
    return r;
}

int cvs_canonicalise(const char *in, size_t n, char **out, size_t *out_n) {
    jv *v = NULL;
    if (parse_doc(in, n, &v)) return CVS_ERR;
    int r = canon_of(v, out, out_n);
    jfree(v);
    return r;
}
int cvs_parse_canonical(const char *in, size_t n) {
    char *c = NULL; size_t cn = 0;
    if (n > MAX_RECORD) return fail("%zu bytes exceeds the %d-byte cap", n, MAX_RECORD);
    if (cvs_canonicalise(in, n, &c, &cn)) return CVS_ERR;
    int same = cn == n && !memcmp(c, in, n);
    free(c);
    return same ? CVS_OK : fail("valid JSON but not in canonical form");
}

/* ================================ primitives ================================ */
void cvs_sha256(const uint8_t *d, size_t n, uint8_t out[32]) { crypto_hash_sha256(out, d, n); }
static void hex(const uint8_t *d, size_t n, char *out) { sodium_bin2hex(out, n * 2 + 1, d, n); }
static int unhex(const char *s, uint8_t *out, size_t n) {
    size_t len = 0;
    if (strlen(s) != n * 2) return CVS_ERR;
    for (size_t i = 0; i < n * 2; i++) if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return CVS_ERR;
    return sodium_hex2bin(out, n, s, n * 2, NULL, &len, NULL) == 0 && len == n ? CVS_OK : CVS_ERR;
}
void cvs_key_id(const uint8_t pub[32], char out[65]) { uint8_t h[32]; cvs_sha256(pub, 32, h); hex(h, 32, out); }
void cvs_keypair_from_seed(const uint8_t seed[32], uint8_t pub[32], uint8_t sk[64]) { crypto_sign_seed_keypair(pub, sk, seed); }
void cvs_leaf_hash(const uint8_t *s, size_t n, uint8_t out[32]) {
    crypto_hash_sha256_state st;
    uint8_t z = 0;
    crypto_hash_sha256_init(&st); crypto_hash_sha256_update(&st, &z, 1); crypto_hash_sha256_update(&st, s, n);
    crypto_hash_sha256_final(&st, out);
}
static void node_hash(const uint8_t *l, const uint8_t *r, uint8_t out[32]) {
    crypto_hash_sha256_state st;
    uint8_t o = 1;
    crypto_hash_sha256_init(&st); crypto_hash_sha256_update(&st, &o, 1);
    crypto_hash_sha256_update(&st, l, 32); crypto_hash_sha256_update(&st, r, 32);
    crypto_hash_sha256_final(&st, out);
}
void cvs_merkle_root(const uint8_t *leaves, size_t n, uint8_t out[32]) {
    if (n == 0) { cvs_sha256((const uint8_t *)"", 0, out); return; }
    if (n == 1) { memcpy(out, leaves, 32); return; }
    size_t k = 1;
    while (k * 2 < n) k *= 2;
    uint8_t l[32], r[32];
    cvs_merkle_root(leaves, k, l);
    cvs_merkle_root(leaves + 32 * k, n - k, r);
    node_hash(l, r, out);
}
static void link_of(const uint8_t rh[32], const uint8_t sig[64], char out[65]) {
    crypto_hash_sha256_state st;
    uint8_t d = 2, h[32];
    crypto_hash_sha256_init(&st); crypto_hash_sha256_update(&st, &d, 1);
    crypto_hash_sha256_update(&st, rh, 32); crypto_hash_sha256_update(&st, sig, 64);
    crypto_hash_sha256_final(&st, h);
    hex(h, 32, out);
}
static int sign_tagged(const char *tag, const char *msg, size_t n, const uint8_t sk[64], uint8_t sig[64]) {
    size_t tl = strlen(tag);
    uint8_t *m = malloc(tl + n);
    if (!m) return fail("out of memory");
    memcpy(m, tag, tl); memcpy(m + tl, msg, n);
    unsigned long long sl = 0;
    int r = crypto_sign_detached(sig, &sl, m, tl + n, sk);
    free(m);
    return r ? fail("signing failed") : CVS_OK;
}
static int verify_tagged(const char *tag, const char *msg, size_t n, const uint8_t sig[64], const uint8_t pub[32]) {
    size_t tl = strlen(tag);
    uint8_t *m = malloc(tl + n);
    if (!m) return 0;
    memcpy(m, tag, tl); memcpy(m + tl, msg, n);
    int ok = crypto_sign_verify_detached(sig, m, tl + n, pub) == 0;
    free(m);
    return ok;
}

/* ================================ one record ================================ */
static int ts_ok(const char *s) {
    if (strlen(s) != 27) return 0;
    for (int i = 0; i < 27; i++) {
        char c = s[i];
        if (i == 4 || i == 7) { if (c != '-') return 0; }
        else if (i == 10) { if (c != 'T') return 0; }
        else if (i == 13 || i == 16) { if (c != ':') return 0; }
        else if (i == 19) { if (c != '.') return 0; }
        else if (i == 26) { if (c != 'Z') return 0; }
        else if (c < '0' || c > '9') return 0;
    }
    return 1;
}
int cvs_seal_record(const char *type, uint64_t seq, const char *prev_hash, const uint8_t pub[32], const uint8_t sk[64],
                    const char *created_at_utc, const uint8_t nonce[16], const char *body_json, char **stored,
                    size_t *stored_n, uint8_t record_hash[32], char link_hex[65]) {
    uint8_t pb[32];
    if (unhex(prev_hash, pb, 32)) return fail("prev_record_hash must be 64 lowercase hex");
    if (!ts_ok(created_at_utc)) return fail("created_at_utc must be YYYY-MM-DDTHH:MM:SS.ffffffZ");
    jv *root = NULL;
    if (parse_doc(body_json, strlen(body_json), &root)) return CVS_ERR;
    char kid[65], nh[33];
    cvs_key_id(pub, kid);
    hex(nonce, 16, nh);
    int r = jpush(root, "v", jstr(CVS_VERSION)) || jpush(root, "type", jstr(type)) || jpush(root, "seq", jint((int64_t)seq)) ||
            jpush(root, "prev_record_hash", jstr(prev_hash)) || jpush(root, "key_id", jstr(kid)) ||
            jpush(root, "created_at_utc", jstr(created_at_utc)) || jpush(root, "nonce", jstr(nh));
    if (r) { jfree(root); return CVS_ERR; }
    /* header names must not collide with body sections */
    for (size_t i = 0; i + 1 < root->n; i++)
        for (size_t j = i + 1; j < root->n; j++)
            if (!strcmp(root->k[i], root->k[j])) { int e = fail("body member \"%s\" collides with a header field", root->k[i]); jfree(root); return e; }
    char *cu = NULL; size_t cn = 0;
    if (canon_unsigned(root, &cu, &cn)) { jfree(root); return CVS_ERR; }
    uint8_t sig[64], rh[32];
    if (sign_tagged(TAG_RECORD, cu, cn, sk, sig)) { free(cu); jfree(root); return CVS_ERR; }
    cvs_sha256((const uint8_t *)cu, cn, rh);
    free(cu);
    char sh[129];
    hex(sig, 64, sh);
    if (jpush(root, "signature", jstr(sh))) { jfree(root); return CVS_ERR; }
    r = canon_of(root, stored, stored_n);
    jfree(root);
    if (r) return CVS_ERR;
    if (*stored_n > MAX_RECORD) { free(*stored); *stored = NULL; return fail("record is %zu bytes, over the %d-byte cap", *stored_n, MAX_RECORD); }
    if (record_hash) memcpy(record_hash, rh, 32);
    if (link_hex) link_of(rh, sig, link_hex);
    return CVS_OK;
}

/* ================================ ledger ================================ */
struct cvs_ledger {
    sqlite3 *db;
    uint8_t seed[32], pub[32], sk[64];
    char key_id[65];
    uint64_t every;
};

static int sq(sqlite3 *db, const char *sql) {
    char *e = NULL;
    if (sqlite3_exec(db, sql, NULL, NULL, &e) != SQLITE_OK) { int r = fail("sqlite: %s", e ? e : "error"); sqlite3_free(e); return r; }
    return CVS_OK;
}
static const char SCHEMA[] =
    "CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);"
    "CREATE TABLE records (seq INTEGER PRIMARY KEY, type TEXT NOT NULL, nonce TEXT NOT NULL UNIQUE, rec_hash BLOB NOT NULL, rec TEXT NOT NULL);"
    "CREATE TABLE merkle_nodes (level INTEGER NOT NULL, idx INTEGER NOT NULL, hash BLOB NOT NULL, PRIMARY KEY (level, idx));"
    "CREATE TABLE payloads (hash BLOB PRIMARY KEY, data BLOB NOT NULL);"
    "CREATE TRIGGER records_no_update BEFORE UPDATE ON records BEGIN SELECT RAISE(ABORT, 'append-only'); END;"
    "CREATE TRIGGER records_no_delete BEFORE DELETE ON records BEGIN SELECT RAISE(ABORT, 'append-only'); END;"
    "CREATE TRIGGER merkle_no_update BEFORE UPDATE ON merkle_nodes BEGIN SELECT RAISE(ABORT, 'append-only'); END;"
    "CREATE TRIGGER merkle_no_delete BEFORE DELETE ON merkle_nodes BEGIN SELECT RAISE(ABORT, 'append-only'); END;"
    "CREATE TRIGGER payloads_no_update BEFORE UPDATE ON payloads BEGIN SELECT RAISE(ABORT, 'append-only'); END;"
    "CREATE TRIGGER payloads_no_delete BEFORE DELETE ON payloads BEGIN SELECT RAISE(ABORT, 'append-only'); END;"
    "CREATE TRIGGER meta_no_update BEFORE UPDATE ON meta BEGIN SELECT RAISE(ABORT, 'append-only'); END;"
    "CREATE TRIGGER meta_no_delete BEFORE DELETE ON meta BEGIN SELECT RAISE(ABORT, 'append-only'); END;";

static int node_get(sqlite3 *db, int64_t level, int64_t idx, uint8_t out[32]) {
    sqlite3_stmt *s;
    if (sqlite3_prepare_v2(db, "SELECT hash FROM merkle_nodes WHERE level=? AND idx=?", -1, &s, NULL)) return fail("sqlite: %s", sqlite3_errmsg(db));
    sqlite3_bind_int64(s, 1, level); sqlite3_bind_int64(s, 2, idx);
    int r = CVS_ERR;
    if (sqlite3_step(s) == SQLITE_ROW && sqlite3_column_bytes(s, 0) == 32) { memcpy(out, sqlite3_column_blob(s, 0), 32); r = CVS_OK; }
    else fail("merkle node (%lld,%lld) missing", (long long)level, (long long)idx);
    sqlite3_finalize(s);
    return r;
}
static int node_put(sqlite3 *db, int64_t level, int64_t idx, const uint8_t h[32]) {
    sqlite3_stmt *s;
    if (sqlite3_prepare_v2(db, "INSERT INTO merkle_nodes(level, idx, hash) VALUES (?,?,?)", -1, &s, NULL)) return fail("sqlite: %s", sqlite3_errmsg(db));
    sqlite3_bind_int64(s, 1, level); sqlite3_bind_int64(s, 2, idx); sqlite3_bind_blob(s, 3, h, 32, SQLITE_STATIC);
    int r = sqlite3_step(s) == SQLITE_DONE ? CVS_OK : fail("sqlite: %s", sqlite3_errmsg(db));
    sqlite3_finalize(s);
    return r;
}
static int tree_append(sqlite3 *db, uint64_t n, const uint8_t leaf[32]) {
    uint8_t cur[32], left[32];
    memcpy(cur, leaf, 32);
    int64_t level = 0, i = (int64_t)n;
    if (node_put(db, level, i, cur)) return CVS_ERR;
    while (i & 1) {
        if (node_get(db, level, i - 1, left)) return CVS_ERR;
        node_hash(left, cur, cur);
        level++; i >>= 1;
        if (node_put(db, level, i, cur)) return CVS_ERR;
    }
    return CVS_OK;
}
/* MTH over the first n leaves from the stored complete-subtree nodes */
static int tree_root(sqlite3 *db, uint64_t n, uint8_t out[32]) {
    if (n == 0) { cvs_sha256((const uint8_t *)"", 0, out); return CVS_OK; }
    uint8_t acc[32], nd[32];
    int have = 0;
    for (int level = 0; level < 63; level++) {
        if (!((n >> level) & 1)) continue;
        if (node_get(db, level, (int64_t)((n >> level) - 1), nd)) return CVS_ERR;
        if (!have) { memcpy(acc, nd, 32); have = 1; }
        else node_hash(nd, acc, acc);
    }
    memcpy(out, acc, 32);
    return CVS_OK;
}
static int meta_get(sqlite3 *db, const char *k, char *out, size_t cap) {
    sqlite3_stmt *s;
    if (sqlite3_prepare_v2(db, "SELECT v FROM meta WHERE k=?", -1, &s, NULL)) return CVS_ERR;
    sqlite3_bind_text(s, 1, k, -1, SQLITE_STATIC);
    int r = CVS_ERR;
    if (sqlite3_step(s) == SQLITE_ROW) { snprintf(out, cap, "%s", (const char *)sqlite3_column_text(s, 0)); r = CVS_OK; }
    sqlite3_finalize(s);
    return r;
}
static void now_utc(char out[28]) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    struct tm tm;
    gmtime_r(&t.tv_sec, &tm);
    size_t n = strftime(out, 28, "%Y-%m-%dT%H:%M:%S", &tm);
    snprintf(out + n, 28 - n, ".%06ldZ", t.tv_nsec / 1000);
}

/* the last record: its seq and link */
static int tip(sqlite3 *db, int64_t *seq, char link[65], int64_t *count) {
    sqlite3_stmt *s;
    if (sqlite3_prepare_v2(db, "SELECT seq, rec FROM records ORDER BY seq DESC LIMIT 1", -1, &s, NULL)) return fail("sqlite: %s", sqlite3_errmsg(db));
    int r = CVS_ERR;
    if (sqlite3_step(s) == SQLITE_ROW) {
        *seq = sqlite3_column_int64(s, 0);
        *count = *seq + 1;
        const char *rec = (const char *)sqlite3_column_text(s, 1);
        jv *o = NULL;
        if (!parse_doc(rec, strlen(rec), &o)) {
            char *cu = NULL; size_t cn = 0;
            const char *sh = jgets(o, "signature");
            uint8_t sig[64], rh[32];
            if (sh && !unhex(sh, sig, 64) && !canon_unsigned(o, &cu, &cn)) {
                cvs_sha256((const uint8_t *)cu, cn, rh);
                link_of(rh, sig, link);
                r = CVS_OK;
            }
            free(cu);
            jfree(o);
        }
        if (r) fail("the tip record is not readable");
    } else { *seq = -1; *count = 0; link[0] = 0; r = CVS_OK; }
    sqlite3_finalize(s);
    return r;
}

static int insert_record(cvs_ledger *l, const char *type, uint64_t seq, const char *prev, const uint8_t pub[32], const uint8_t sk[64],
                         const char *now, const uint8_t nonce[16], const char *body, char link_out[65]) {
    char *st = NULL; size_t sn = 0; uint8_t rh[32], leaf[32];
    if (cvs_seal_record(type, seq, prev, pub, sk, now, nonce, body, &st, &sn, rh, link_out)) return CVS_ERR;
    char nh[33];
    hex(nonce, 16, nh);
    sqlite3_stmt *s;
    if (sqlite3_prepare_v2(l->db, "INSERT INTO records(seq, type, nonce, rec_hash, rec) VALUES (?,?,?,?,?)", -1, &s, NULL)) { free(st); return fail("sqlite: %s", sqlite3_errmsg(l->db)); }
    sqlite3_bind_int64(s, 1, (int64_t)seq); sqlite3_bind_text(s, 2, type, -1, SQLITE_STATIC); sqlite3_bind_text(s, 3, nh, -1, SQLITE_STATIC);
    sqlite3_bind_blob(s, 4, rh, 32, SQLITE_STATIC); sqlite3_bind_text(s, 5, st, (int)sn, SQLITE_STATIC);
    int r = sqlite3_step(s) == SQLITE_DONE ? CVS_OK : fail("sqlite: %s", sqlite3_errmsg(l->db));
    sqlite3_finalize(s);
    if (!r) { cvs_leaf_hash((const uint8_t *)st, sn, leaf); r = tree_append(l->db, seq, leaf); }
    free(st);
    return r;
}
static void derive_nonce(const uint8_t *base, uint64_t seq, uint8_t out[16]) {
    if (!base) { randombytes_buf(out, 16); return; }
    uint8_t buf2[16 + 8 + 4], h[32];
    memcpy(buf2, base, 16);
    for (int i = 0; i < 8; i++) buf2[16 + i] = (uint8_t)(seq >> (56 - 8 * i));
    memcpy(buf2 + 24, "ckpt", 4);
    cvs_sha256(buf2, sizeof buf2, h);
    memcpy(out, h, 16);
}

int cvs_ledger_init(const char *path, const uint8_t seed[32], const char *manifest_json, cvs_ledger **out) {
    return cvs_ledger_init_at(path, seed, manifest_json, NULL, NULL, out);
}
int cvs_ledger_init_at(const char *path, const uint8_t seed[32], const char *manifest_json, const char *now,
                       const uint8_t *nonce_in, cvs_ledger **out) {
    if (cvs_init()) return CVS_ERR;
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return fail("%s already exists; refusing to re-initialise a ledger", path); }
    cvs_ledger *l = calloc(1, sizeof *l);
    if (!l) return fail("out of memory");
    memcpy(l->seed, seed, 32);
    cvs_keypair_from_seed(seed, l->pub, l->sk);
    cvs_key_id(l->pub, l->key_id);
    jv *m = NULL;
    if (parse_doc(manifest_json, strlen(manifest_json), &m)) { free(l); return CVS_ERR; }
    if (!jget(m, "spec") && jpush(m, "spec", jstr(CVS_VERSION))) { jfree(m); free(l); return CVS_ERR; }
    if (jget(m, "key_id")) { jfree(m); free(l); return fail("the manifest's key_id is set by the ledger, not the caller"); }
    if (jpush(m, "key_id", jstr(l->key_id))) { jfree(m); free(l); return CVS_ERR; }
    jv *ce = jget(m, "checkpoint_every");
    if (!ce || ce->t != J_INT || ce->i < 1) { jfree(m); free(l); return fail("manifest needs checkpoint_every >= 1"); }
    l->every = (uint64_t)ce->i;
    char *mc = NULL; size_t mn = 0;
    if (canon_of(m, &mc, &mn)) { jfree(m); free(l); return CVS_ERR; }
    jfree(m);
    /* genesis prev_record_hash = SHA-256(0x03 || canon(manifest)) */
    uint8_t *tmp = malloc(mn + 1);
    tmp[0] = 3; memcpy(tmp + 1, mc, mn);
    uint8_t mh[32]; char mhx[65];
    cvs_sha256(tmp, mn + 1, mh); hex(mh, 32, mhx);
    free(tmp);
    if (sqlite3_open_v2(path, &l->db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL)) { free(mc); int r = fail("cannot create %s", path); free(l); return r; }
    sqlite3_busy_timeout(l->db, 5000);
    char every[32], body[4096];
    snprintf(every, sizeof every, "%llu", (unsigned long long)l->every);
    int r = sq(l->db, "PRAGMA journal_mode=WAL") || sq(l->db, "PRAGMA synchronous=FULL") || sq(l->db, SCHEMA);
    if (!r) {
        sqlite3_stmt *s;
        const char *kv[][2] = {{"store_version", "cva-seal-store/2"}, {"durability", "per_record"}, {"group_n", "100"},
                               {"group_ms", "50"}, {"checkpoint_every", every}, {"loss_window", "0 records"}};
        for (size_t i = 0; !r && i < 6; i++) {
            sqlite3_prepare_v2(l->db, "INSERT INTO meta(k, v) VALUES (?,?)", -1, &s, NULL);
            sqlite3_bind_text(s, 1, kv[i][0], -1, SQLITE_STATIC); sqlite3_bind_text(s, 2, kv[i][1], -1, SQLITE_STATIC);
            r = sqlite3_step(s) == SQLITE_DONE ? CVS_OK : fail("sqlite: %s", sqlite3_errmsg(l->db));
            sqlite3_finalize(s);
        }
    }
    if (!r) {
        snprintf(body, sizeof body, "{\"deployment_manifest\":%s}", mc);
        char nowb[28]; now_utc(nowb);
        uint8_t nonce[16];
        if (nonce_in) memcpy(nonce, nonce_in, 16); else randombytes_buf(nonce, 16);
        char lk[65];
        r = sq(l->db, "BEGIN IMMEDIATE") || insert_record(l, "genesis", 0, mhx, l->pub, l->sk, now ? now : nowb, nonce, body, lk) || sq(l->db, "COMMIT");
    }
    free(mc);
    if (r) { sqlite3_close(l->db); free(l); remove(path); return CVS_ERR; }
    *out = l;
    return CVS_OK;
}

static int active_key_id(sqlite3 *db, char out[65]) {
    sqlite3_stmt *s;
    const char *q = "SELECT rec FROM records WHERE type='key_rotation' ORDER BY seq DESC LIMIT 1";
    if (sqlite3_prepare_v2(db, q, -1, &s, NULL)) return fail("sqlite: %s", sqlite3_errmsg(db));
    int r = CVS_ERR;
    if (sqlite3_step(s) == SQLITE_ROW) {
        const char *rec = (const char *)sqlite3_column_text(s, 0);
        jv *o = NULL;
        if (!parse_doc(rec, strlen(rec), &o)) {
            const char *k = jgets(jget(o, "rotation"), "new_key_id");
            if (k) { snprintf(out, 65, "%s", k); r = CVS_OK; }
            jfree(o);
        }
    } else {
        sqlite3_finalize(s);
        if (sqlite3_prepare_v2(db, "SELECT rec FROM records WHERE seq=0", -1, &s, NULL)) return fail("sqlite: %s", sqlite3_errmsg(db));
        if (sqlite3_step(s) == SQLITE_ROW) {
            const char *rec = (const char *)sqlite3_column_text(s, 0);
            jv *o = NULL;
            if (!parse_doc(rec, strlen(rec), &o)) { const char *k = jgets(o, "key_id"); if (k) { snprintf(out, 65, "%s", k); r = CVS_OK; } jfree(o); }
        }
    }
    sqlite3_finalize(s);
    return r ? fail("cannot determine the ledger's active key") : CVS_OK;
}

int cvs_ledger_open(const char *path, const uint8_t seed[32], cvs_ledger **out) {
    if (cvs_init()) return CVS_ERR;
    cvs_ledger *l = calloc(1, sizeof *l);
    if (!l) return fail("out of memory");
    memcpy(l->seed, seed, 32);
    cvs_keypair_from_seed(seed, l->pub, l->sk);
    cvs_key_id(l->pub, l->key_id);
    if (sqlite3_open_v2(path, &l->db, SQLITE_OPEN_READWRITE, NULL)) { free(l); return fail("cannot open %s", path); }
    sqlite3_busy_timeout(l->db, 5000);
    char v[64], ev[32], act[65];
    if (meta_get(l->db, "store_version", v, sizeof v) || strcmp(v, "cva-seal-store/2")) { sqlite3_close(l->db); free(l); return fail("%s is not a cva-seal-store/2 ledger", path); }
    if (meta_get(l->db, "checkpoint_every", ev, sizeof ev)) { sqlite3_close(l->db); free(l); return fail("ledger has no checkpoint_every"); }
    l->every = strtoull(ev, NULL, 10);
    if (active_key_id(l->db, act)) { sqlite3_close(l->db); free(l); return CVS_ERR; }
    if (strcmp(act, l->key_id)) {
        char mine[17];
        snprintf(mine, sizeof mine, "%.16s", l->key_id);
        sqlite3_close(l->db); free(l);
        return fail("key %s… is not this ledger's active signing key (%.16s…)", mine, act);
    }
    sq(l->db, "PRAGMA synchronous=FULL");
    *out = l;
    return CVS_OK;
}
void cvs_ledger_close(cvs_ledger *l) { if (l) { sqlite3_close(l->db); sodium_memzero(l->seed, 32); sodium_memzero(l->sk, 64); free(l); } }
uint64_t cvs_ledger_size(cvs_ledger *l) { int64_t s, c; char lk[65]; return tip(l->db, &s, lk, &c) ? 0 : (uint64_t)c; }

int cvs_ledger_put_payload(cvs_ledger *l, const uint8_t *data, size_t n, char address[65]) {
    uint8_t h[32];
    cvs_sha256(data, n, h);
    hex(h, 32, address);
    sqlite3_stmt *s;
    if (sqlite3_prepare_v2(l->db, "INSERT OR IGNORE INTO payloads(hash, data) VALUES (?,?)", -1, &s, NULL)) return fail("sqlite: %s", sqlite3_errmsg(l->db));
    sqlite3_bind_blob(s, 1, h, 32, SQLITE_STATIC); sqlite3_bind_blob(s, 2, data, (int)n, SQLITE_STATIC);
    int r = sqlite3_step(s) == SQLITE_DONE ? CVS_OK : fail("sqlite: %s", sqlite3_errmsg(l->db));
    sqlite3_finalize(s);
    return r;
}

/* append one record (+ a checkpoint if the cadence says so) inside a transaction; `key`/`sk` sign it */
static int append_tx(cvs_ledger *l, const char *type, const char *body, const char *now, const uint8_t *nonce,
                     const uint8_t pub[32], const uint8_t sk[64], uint64_t *seq_out) {
    if (sq(l->db, "BEGIN IMMEDIATE")) return CVS_ERR;
    int64_t seq, count;
    char prev[65], lk[65], nowb[28];
    uint8_t n1[16];
    int r = tip(l->db, &seq, prev, &count);
    if (!r && count == 0) r = fail("no genesis record");
    if (!r) {
        if (!now) { now_utc(nowb); now = nowb; }
        if (nonce) memcpy(n1, nonce, 16); else randombytes_buf(n1, 16);
        r = insert_record(l, type, (uint64_t)count, prev, pub, sk, now, n1, body, lk);
    }
    if (!r) {
        if (seq_out) *seq_out = (uint64_t)count;
        count++;
        if (strcmp(type, "checkpoint") && strcmp(type, "genesis") && (uint64_t)count % l->every == 0) {
            uint8_t root[32]; char rx[65], cb[256], n2[16];
            r = tree_root(l->db, (uint64_t)count, root);
            if (!r) {
                hex(root, 32, rx);
                snprintf(cb, sizeof cb, "{\"checkpoint\":{\"root_hash\":\"%s\",\"tree_size\":%llu}}", rx, (unsigned long long)count);
                derive_nonce(nonce, (uint64_t)count, (uint8_t *)n2);
                char next[65];
                r = insert_record(l, "checkpoint", (uint64_t)count, lk, l->pub, l->sk, now, (uint8_t *)n2, cb, next);
                (void)next;
            }
        }
    }
    if (r) { sqlite3_exec(l->db, "ROLLBACK", NULL, NULL, NULL); return CVS_ERR; }
    return sq(l->db, "COMMIT");
}
/* analyst_event rules (spec section 4.2), enforced at the SDK: an override without a recorded justification is how
 * assurance systems fail, so it never reaches the chain. Other sections are validated by the Python verifier. */
static int in_set(const char *v, const char *const *set) { for (; *set; set++) if (!strcmp(v, *set)) return 1; return 0; }
static int is_hexn(const char *s, size_t n) {
    if (!s || strlen(s) != n) return 0;
    for (; *s; s++) if (!((*s >= '0' && *s <= '9') || (*s >= 'a' && *s <= 'f'))) return 0;
    return 1;
}
static int validate_analyst(const jv *body) {
    static const char *const req[] = {"actor_id", "role", "action", "scan_id", "target_type", "target_ref", "finding_id", "justification", "request_id", NULL};
    static const char *const opt[] = {"new_disposition", "reason_code", "assignee", "refs_seq", "expected_prev_seq", NULL};
    static const char *const actions[] = {"assign", "acknowledge", "override", "approve", "quarantine", "release", NULL};
    static const char *const targets[] = {"sample", "contributor", "batch", "model", "record", "dataset", NULL};
    static const char *const disp[] = {"accept", "review", "quarantine", NULL};
    static const char *const reasons[] = {"quality_issue", "known_benign", "insufficient_evidence", "accepted_risk", "superseded_by_rescan",
        "other_with_justification", "known_restore", "test_data", "superseded_ledger", "false_positive_confirmed", NULL};
    if (body->n != 1 || !body->k || strcmp(body->k[0], "analyst")) return fail("an analyst_event has exactly one section, 'analyst'");
    const jv *a = body->v[0];
    if (a->t != J_OBJ) return fail("analyst must be an object");
    for (const char *const *k = req; *k; k++) if (!jget(a, *k)) return fail("analyst.%s is required", *k);
    for (size_t i = 0; i < a->n; i++) if (!in_set(a->k[i], req) && !in_set(a->k[i], opt)) return fail("analyst.%s is not a known field", a->k[i]);
    for (const char *const *k = req; *k; k++) if (jget(a, *k)->t != J_STR) return fail("analyst.%s must be a string", *k);
    const char *action = jgets(a, "action");
    if (!in_set(action, actions)) return fail("analyst.action is not one of assign/acknowledge/override/approve/quarantine/release");
    if (!in_set(jgets(a, "target_type"), targets)) return fail("analyst.target_type is not a known target type");
    if (!is_hexn(jgets(a, "request_id"), 32)) return fail("analyst.request_id must be 32 lowercase hex characters");
    if (jget(a, "new_disposition") && !(jgets(a, "new_disposition") && in_set(jgets(a, "new_disposition"), disp))) return fail("analyst.new_disposition is not accept/review/quarantine");
    if (jget(a, "reason_code") && !(jgets(a, "reason_code") && in_set(jgets(a, "reason_code"), reasons))) return fail("analyst.reason_code is not in the closed set");
    for (const char *k2 = "refs_seq"; k2; k2 = !strcmp(k2, "refs_seq") ? "expected_prev_seq" : NULL) {
        const jv *x = jget(a, k2);
        if (x && (x->t != J_INT || x->i < 0)) return fail("analyst.%s must be a non-negative integer", k2);
    }
    if (!strcmp(action, "override")) {
        if (!jget(a, "new_disposition") || !jget(a, "reason_code") || !jget(a, "expected_prev_seq")) return fail("an override needs new_disposition, reason_code and expected_prev_seq");
        const char *j = jgets(a, "justification");
        size_t k = 0;
        while (j[k] == ' ') k++;
        if (!j[k]) return fail("an override needs a recorded justification: it is empty");
    } else if (!strcmp(action, "approve") && !jget(a, "refs_seq")) return fail("an approval must name the override it approves (refs_seq)");
    else if (!strcmp(action, "assign") && !jget(a, "assignee")) return fail("an assignment must name the assignee");
    return CVS_OK;
}
int cvs_ledger_append(cvs_ledger *l, const char *type, const char *body_json, const char *now, const uint8_t *nonce, uint64_t *seq_out) {
    if (!strcmp(type, "analyst_event")) {
        jv *b = NULL;
        if (parse_doc(body_json, strlen(body_json), &b)) return CVS_ERR;
        int r = validate_analyst(b);
        jfree(b);
        if (r) return CVS_ERR;
    }
    if (!strcmp(type, "genesis")) return fail("genesis is written by cvs_ledger_init only");
    if (!strcmp(type, "key_rotation")) return fail("use cvs_ledger_rotate for key rotations");
    if (!strcmp(type, "checkpoint")) return fail("checkpoints are written on the cadence, not by callers");
    return append_tx(l, type, body_json, now, nonce, l->pub, l->sk, seq_out);
}
int cvs_ledger_rotate(cvs_ledger *l, const uint8_t new_seed[32], const char *now, const uint8_t *nonce) {
    uint8_t npub[32], nsk[64];
    cvs_keypair_from_seed(new_seed, npub, nsk);
    char nkid[65], npx[65];
    cvs_key_id(npub, nkid); hex(npub, 32, npx);
    if (!strcmp(nkid, l->key_id)) return fail("cannot rotate to the key that is already active");
    int64_t seq, count; char lk[65];
    if (tip(l->db, &seq, lk, &count)) return CVS_ERR;
    uint64_t eff = (uint64_t)count + 1;                  /* the rotation sits at seq == count; effective_seq = seq + 1 */
    char popm[256], pb[512];
    snprintf(popm, sizeof popm, "{\"effective_seq\":%llu,\"new_key_id\":\"%s\",\"prev_key_id\":\"%s\"}", (unsigned long long)eff, nkid, l->key_id);
    uint8_t sig[64];
    if (sign_tagged(TAG_POP, popm, strlen(popm), nsk, sig)) return CVS_ERR;
    char sx[129];
    hex(sig, 64, sx);
    snprintf(pb, sizeof pb, "{\"rotation\":{\"effective_seq\":%llu,\"new_key_id\":\"%s\",\"new_key_pop\":\"%s\",\"new_public_key\":\"%s\"}}",
             (unsigned long long)eff, nkid, sx, npx);
    /* the rotation is signed by the OUTGOING key; the checkpoint it may trigger by the NEW key (effective_seq) */
    if (sq(l->db, "BEGIN IMMEDIATE")) return CVS_ERR;
    char prev[65], lk2[65], nowb[28];
    uint8_t n1[16];
    int r = tip(l->db, &seq, prev, &count);
    if (!r) {
        if (!now) { now_utc(nowb); now = nowb; }
        if (nonce) memcpy(n1, nonce, 16); else randombytes_buf(n1, 16);
        r = insert_record(l, "key_rotation", (uint64_t)count, prev, l->pub, l->sk, now, n1, pb, lk2);
    }
    if (!r) {
        count++;
        if ((uint64_t)count % l->every == 0) {
            uint8_t root[32], n2[16]; char rx[65], cb[256], nx[65];
            r = tree_root(l->db, (uint64_t)count, root);
            if (!r) {
                hex(root, 32, rx);
                snprintf(cb, sizeof cb, "{\"checkpoint\":{\"root_hash\":\"%s\",\"tree_size\":%llu}}", rx, (unsigned long long)count);
                derive_nonce(nonce, (uint64_t)count, n2);
                r = insert_record(l, "checkpoint", (uint64_t)count, lk2, npub, nsk, now, n2, cb, nx);
            }
        }
    }
    if (r) { sqlite3_exec(l->db, "ROLLBACK", NULL, NULL, NULL); return CVS_ERR; }
    if (sq(l->db, "COMMIT")) return CVS_ERR;
    memcpy(l->seed, new_seed, 32); memcpy(l->pub, npub, 32); memcpy(l->sk, nsk, 64);
    memcpy(l->key_id, nkid, 65);
    return CVS_OK;
}

/* ================================ verification ================================ */
#define MAXLEAVES_INIT 1024
static int verify_records(char **recs, size_t *lens, size_t n, const uint8_t gpub[32], const char *expect, uint64_t *count_out) {
    uint8_t (*leaves)[32] = malloc((n ? n : 1) * 32);
    uint8_t pub[32];
    char active[65], prev_link[65] = {0};
    if (!leaves) return fail("out of memory");
    memcpy(pub, gpub, 32);
    cvs_key_id(gpub, active);
    int r = CVS_OK;
    for (size_t i = 0; i < n && !r; i++) {
        if (cvs_parse_canonical(recs[i], lens[i])) { char m[512]; snprintf(m, sizeof m, "%s", g_err); r = fail("record at position %zu: %s", i, m); break; }
        jv *o = NULL;
        parse_doc(recs[i], lens[i], &o);
        const char *type = jgets(o, "type"), *kid = jgets(o, "key_id"), *prv = jgets(o, "prev_record_hash"), *sh = jgets(o, "signature");
        jv *sq_ = jget(o, "seq");
        uint8_t sig[64];
        if (!type || !kid || !prv || !sh || !sq_ || sq_->t != J_INT || unhex(sh, sig, 64)) { r = fail("record %zu: malformed header", i); jfree(o); break; }
        if (sq_->i != (int64_t)i) { r = fail("record %zu: seq %lld is not its position", i, (long long)sq_->i); jfree(o); break; }
        if ((i == 0) != !strcmp(type, "genesis")) { r = fail("record %zu: genesis must be first and only first", i); jfree(o); break; }
        if (strcmp(kid, active)) { r = fail("record %zu: signed by key %.16s…, not the active key %.16s…", i, kid, active); jfree(o); break; }
        char *cu = NULL; size_t cn = 0;
        if (canon_unsigned(o, &cu, &cn)) { r = CVS_ERR; jfree(o); break; }
        if (!verify_tagged(TAG_RECORD, cu, cn, sig, pub)) { r = fail("record %zu: signature does not verify", i); free(cu); jfree(o); break; }
        uint8_t rh[32];
        cvs_sha256((const uint8_t *)cu, cn, rh);
        free(cu);
        if (i == 0) {
            jv *man = jget(o, "deployment_manifest");
            char *mc = NULL; size_t mn = 0;
            if (!man || canon_of(man, &mc, &mn)) { r = fail("genesis has no manifest"); jfree(o); break; }
            uint8_t *t = malloc(mn + 1); t[0] = 3; memcpy(t + 1, mc, mn);
            uint8_t mh[32]; char mx[65];
            cvs_sha256(t, mn + 1, mh); hex(mh, 32, mx);
            free(t); free(mc);
            if (strcmp(prv, mx)) r = fail("genesis does not commit to its manifest");
            else if (strcmp(jgets(man, "key_id") ? jgets(man, "key_id") : "", kid)) r = fail("genesis key does not match its manifest");
            else if (expect && strcmp(expect, prv)) r = fail("genesis commits to a different deployment than expected");
        } else if (strcmp(prv, prev_link)) r = fail("record %zu: prev_record_hash does not match the previous record's link", i);
        if (!r && !strcmp(type, "key_rotation")) {
            jv *rot = jget(o, "rotation");
            const char *npx = jgets(rot, "new_public_key"), *nk = jgets(rot, "new_key_id"), *pp = jgets(rot, "new_key_pop");
            jv *eff = jget(rot, "effective_seq");
            uint8_t np[32], ps[64];
            char calc[65];
            if (!npx || !nk || !pp || !eff || eff->t != J_INT || unhex(npx, np, 32) || unhex(pp, ps, 64)) r = fail("record %zu: malformed rotation", i);
            else {
                cvs_key_id(np, calc);
                char pm[256];
                snprintf(pm, sizeof pm, "{\"effective_seq\":%lld,\"new_key_id\":\"%s\",\"prev_key_id\":\"%s\"}", (long long)eff->i, nk, kid);
                if (strcmp(calc, nk)) r = fail("record %zu: new_key_id is not the hash of new_public_key", i);
                else if (eff->i != (int64_t)i + 1) r = fail("record %zu: effective_seq must be seq + 1", i);
                else if (!verify_tagged(TAG_POP, pm, strlen(pm), ps, np)) r = fail("record %zu: the incoming key's proof of possession does not verify", i);
                else { memcpy(pub, np, 32); snprintf(active, 65, "%s", nk); }
            }
        }
        if (!r && !strcmp(type, "checkpoint")) {
            jv *cp = jget(o, "checkpoint");
            jv *ts = jget(cp, "tree_size");
            const char *rx = jgets(cp, "root_hash");
            uint8_t root[32]; char rxx[65];
            cvs_merkle_root((const uint8_t *)leaves, i, root);
            hex(root, 32, rxx);
            if (!ts || ts->t != J_INT || ts->i != (int64_t)i || !rx || strcmp(rx, rxx)) r = fail("record %zu: checkpoint does not match the records before it", i);
        }
        link_of(rh, sig, prev_link);
        cvs_leaf_hash((const uint8_t *)recs[i], lens[i], leaves[i]);
        jfree(o);
    }
    if (!r && n == 0) r = fail("empty ledger");
    if (count_out) *count_out = n;
    free(leaves);
    return r;
}

int cvs_verify_file(const char *path, const char *genesis_pubkey_hex, const char *expect, uint64_t *records_out) {
    if (cvs_init()) return CVS_ERR;
    uint8_t gpub[32];
    if (unhex(genesis_pubkey_hex, gpub, 32)) return fail("the ledger key must be 64 lowercase hex characters");
    FILE *f = fopen(path, "rb");
    if (!f) return fail("cannot open %s", path);
    char head[16] = {0};
    size_t hn = fread(head, 1, sizeof head, f);
    fclose(f);
    char **recs = NULL; size_t *lens = NULL, n = 0, cap = 0;
    int r = CVS_OK;
    if (hn >= 15 && !memcmp(head, "SQLite format 3", 15)) {
        sqlite3 *db;
        if (sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, NULL)) return fail("cannot open %s", path);
        sqlite3_stmt *s;
        sqlite3_prepare_v2(db, "SELECT rec FROM records ORDER BY rowid", -1, &s, NULL);
        while (sqlite3_step(s) == SQLITE_ROW) {
            if (n == cap) { cap = cap ? cap * 2 : 64; recs = realloc(recs, cap * sizeof *recs); lens = realloc(lens, cap * sizeof *lens); }
            lens[n] = (size_t)sqlite3_column_bytes(s, 0);
            recs[n] = malloc(lens[n] + 1);
            memcpy(recs[n], sqlite3_column_text(s, 0), lens[n]);
            n++;
        }
        sqlite3_finalize(s);
        sqlite3_close(db);
    } else {
        f = fopen(path, "rb");
        buf all = {0};
        char chunk[65536];
        size_t k;
        while ((k = fread(chunk, 1, sizeof chunk, f)) > 0) b_put(&all, chunk, k);
        fclose(f);
        if (all.n && all.p[all.n - 1] != '\n') r = fail("the last line is not terminated by a newline");
        size_t start = 0;
        for (size_t i = 0; !r && i < all.n; i++) {
            if (all.p[i] != '\n') continue;
            if (i == start) { r = fail("blank line at position %zu", n); break; }
            if (n == cap) { cap = cap ? cap * 2 : 64; recs = realloc(recs, cap * sizeof *recs); lens = realloc(lens, cap * sizeof *lens); }
            lens[n] = i - start;
            recs[n] = malloc(lens[n] + 1);
            memcpy(recs[n], all.p + start, lens[n]);
            n++;
            start = i + 1;
        }
        free(all.p);
    }
    if (!r) r = verify_records(recs, lens, n, gpub, expect, records_out);
    for (size_t i = 0; i < n; i++) free(recs[i]);
    free(recs); free(lens);
    return r;
}

/* ================================ tools used by the CLI and the differential tests ================================ */
int cvs_export_file(const char *path) {
    sqlite3 *db;
    if (sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, NULL)) return fail("cannot open %s", path);
    sqlite3_stmt *s;
    if (sqlite3_prepare_v2(db, "SELECT rec FROM records ORDER BY rowid", -1, &s, NULL)) { sqlite3_close(db); return fail("not a ledger"); }
    while (sqlite3_step(s) == SQLITE_ROW) {
        fwrite(sqlite3_column_text(s, 0), 1, (size_t)sqlite3_column_bytes(s, 0), stdout);
        fputc('\n', stdout);
    }
    sqlite3_finalize(s);
    sqlite3_close(db);
    return CVS_OK;
}
static char *read_all(FILE *f, size_t *n) {
    buf b = {0};
    char chunk[65536];
    size_t k;
    while ((k = fread(chunk, 1, sizeof chunk, f)) > 0) if (b_put(&b, chunk, k)) return NULL;
    if (!b.p) b.p = strdup("");
    *n = b.n;
    return b.p;
}
int cvs_roots_stream(FILE *in, FILE *out) {
    size_t n; char *all = read_all(in, &n);
    if (!all) return CVS_ERR;
    size_t cap = 64, count = 0, start = 0;
    uint8_t (*leaves)[32] = malloc(cap * 32);
    for (size_t i = 0; i < n; i++) {
        if (all[i] != '\n') continue;
        if (count == cap) { cap *= 2; leaves = realloc(leaves, cap * 32); }
        cvs_leaf_hash((const uint8_t *)all + start, i - start, leaves[count++]);
        start = i + 1;
        uint8_t root[32]; char rx[65];
        cvs_merkle_root((const uint8_t *)leaves, count, root);
        hex(root, 32, rx);
        fprintf(out, "%zu %s\n", count, rx);
    }
    free(leaves); free(all);
    return CVS_OK;
}
int cvs_reseal_stream(const char *seeds_file, FILE *in, FILE *out) {
    if (cvs_init()) return CVS_ERR;
    FILE *sf = fopen(seeds_file, "rb");
    if (!sf) return fail("cannot read %s", seeds_file);
    struct { uint8_t pub[32], sk[64]; char kid[65]; } keys[16];
    size_t nk = 0;
    char line[128];
    while (nk < 16 && fgets(line, sizeof line, sf)) {
        line[strcspn(line, "\r\n")] = 0;
        uint8_t seed[32];
        if (strlen(line) != 64 || unhex(line, seed, 32)) continue;
        cvs_keypair_from_seed(seed, keys[nk].pub, keys[nk].sk);
        cvs_key_id(keys[nk].pub, keys[nk].kid);
        nk++;
    }
    fclose(sf);
    size_t n; char *all = read_all(in, &n);
    if (!all) return CVS_ERR;
    char prev[65] = {0};
    size_t start = 0, pos = 0;
    for (size_t i = 0; i < n; i++) {
        if (all[i] != '\n') continue;
        jv *o = NULL;
        if (parse_doc(all + start, i - start, &o)) return fail("line %zu: %s", pos, g_err);
        const char *type = jgets(o, "type"), *kid = jgets(o, "key_id"), *ts = jgets(o, "created_at_utc"), *nh = jgets(o, "nonce");
        jv *seqv = jget(o, "seq");
        uint8_t nonce[16];
        if (!type || !kid || !ts || !nh || !seqv || unhex(nh, nonce, 16)) { jfree(o); return fail("line %zu: malformed header", pos); }
        size_t ki = 0;
        while (ki < nk && strcmp(keys[ki].kid, kid)) ki++;
        if (ki == nk) { jfree(o); return fail("line %zu: no seed for key %.16s…", pos, kid); }
        char pv[65];
        if (pos == 0) {
            jv *man = jget(o, "deployment_manifest");
            char *mc = NULL; size_t mn;
            if (!man || canon_of(man, &mc, &mn)) return fail("genesis has no manifest");
            uint8_t *t = malloc(mn + 1); t[0] = 3; memcpy(t + 1, mc, mn);
            uint8_t mh[32];
            cvs_sha256(t, mn + 1, mh); hex(mh, 32, pv);
            free(t); free(mc);
        } else snprintf(pv, sizeof pv, "%s", prev);
        jv *body = jnew(J_OBJ);
        for (size_t m = 0; m < o->n; m++) {
            const char *k = o->k[m];
            if (!strcmp(k, "v") || !strcmp(k, "type") || !strcmp(k, "seq") || !strcmp(k, "prev_record_hash") || !strcmp(k, "key_id") ||
                !strcmp(k, "created_at_utc") || !strcmp(k, "nonce") || !strcmp(k, "signature")) continue;
            /* move, not copy: detach from o */
            jpush(body, k, o->v[m]);
            o->v[m] = jnew(J_NULL);
        }
        char *bj = NULL; size_t bn;
        if (canon_of(body, &bj, &bn)) return CVS_ERR;
        jfree(body);
        char *st = NULL; size_t sn; uint8_t rh[32];
        /* the record's own deployment_manifest was moved out of `o`; genesis body carries it in bj */
        if (cvs_seal_record(type, (uint64_t)seqv->i, pv, keys[ki].pub, keys[ki].sk, ts, nonce, bj, &st, &sn, rh, prev)) return CVS_ERR;
        fwrite(st, 1, sn, out);
        fputc('\n', out);
        free(st); free(bj); jfree(o);
        start = i + 1;
        pos++;
    }
    free(all);
    return CVS_OK;
}
