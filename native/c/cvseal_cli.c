/* cvseal — command line for the C core (gate C9). Used by the differential tests and by operators without Python. */
#include "cvseal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(void) {
    fputs("usage:\n"
          "  cvseal canon                              JSON on stdin -> canonical bytes on stdout\n"
          "  SEED is @FILE, env:NAME or (discouraged) 64 hex chars\n  cvseal keyid SEED                      print the key id and public key\n"
          "  cvseal init DB SEED MANIFEST_JSON [TIME [NONCEHEX]]   create a ledger (genesis)\n"
          "  cvseal append DB SEED TYPE BODY_JSON [TIME [NONCEHEX]]\n"
          "  cvseal payload DB SEED FILE            store a payload, print its address\n"
          "  cvseal rotate DB SEED NEW_SEED [TIME [NONCEHEX]]\n"
          "  cvseal verify FILE PUBKEYHEX [MANIFEST_HASH]   verify a ledger (SQLite or JSONL export)\n"
          "  cvseal export DB                          print the strict JSONL export\n"
          "  cvseal reseal SEEDS_FILE                  re-sign and re-chain a JSONL export from stdin (byte-identity check)\n"
          "  cvseal roots                              JSONL export on stdin -> Merkle root of every prefix\n",
          stderr);
    return 64;
}
static int nib(char c) { return c >= '0' && c <= '9' ? c - '0' : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1); }
/* strict: exactly n bytes of lowercase hex — no whitespace, no sign, nothing sscanf would forgive */
static int hexn(const char *h, uint8_t *out, size_t n) {
    if (strlen(h) != n * 2) return 1;
    for (size_t i = 0; i < n; i++) {
        int a = nib(h[2 * i]), b = nib(h[2 * i + 1]);
        if (a < 0 || b < 0) return 1;
        out[i] = (uint8_t)(a * 16 + b);
    }
    return 0;
}
static int hex16(const char *h, uint8_t out[16]) { return hexn(h, out, 16); }
/* A signing seed. `@FILE` reads it from a file and `env:NAME` from the environment — an argument on the command line is
 * visible to every user in ps and /proc and lands in shell history, so a bare hex seed still works (tests, throwaway
 * ledgers) but prints a warning. */
static int hex32(const char *h, uint8_t out[32]) {
    char buf[128];
    if (h[0] == '@') {
        FILE *f = fopen(h + 1, "rb");
        if (!f) { fprintf(stderr, "cvseal: cannot read seed file %s\n", h + 1); return 1; }
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        buf[n] = 0;
        buf[strcspn(buf, "\r\n")] = 0;
        int r = hexn(buf, out, 32);
        memset(buf, 0, sizeof buf);
        return r;
    }
    if (!strncmp(h, "env:", 4)) {
        const char *v = getenv(h + 4);
        return v ? hexn(v, out, 32) : 1;
    }
    fputs("cvseal: warning: a seed given on the command line is visible in ps and shell history; use @FILE or env:NAME\n", stderr);
    return hexn(h, out, 32);
}
static char *slurp(FILE *f, size_t *n) {
    size_t cap = 1 << 16, len = 0;
    char *b = malloc(cap);
    if (!b) { fputs("cvseal: out of memory\n", stderr); exit(1); }
    size_t k;
    while ((k = fread(b + len, 1, cap - len, f)) > 0) {
        len += k;
        if (len == cap) {
            char *nb = realloc(b, cap * 2);
            if (!nb) { free(b); fputs("cvseal: out of memory\n", stderr); exit(1); }
            b = nb; cap *= 2;
        }
    }
    *n = len;
    return b;
}
static int err(void) { fprintf(stderr, "cvseal: %s\n", cvs_errmsg()); return 1; }

int main(int argc, char **argv) {
    if (argc < 2) return usage();
    if (cvs_init()) return err();
    const char *c = argv[1];
    uint8_t seed[32], nonce[16];
    if (!strcmp(c, "canon")) {
        size_t n; char *in = slurp(stdin, &n), *out = NULL; size_t on;
        if (cvs_canonicalise(in, n, &out, &on)) return err();
        fwrite(out, 1, on, stdout);
        return 0;
    }
    if (!strcmp(c, "keyid") && argc == 3) {
        uint8_t pub[32], sk[64]; char kid[65];
        if (hex32(argv[2], seed)) return usage();
        cvs_keypair_from_seed(seed, pub, sk); cvs_key_id(pub, kid);
        printf("%s ", kid);
        for (int i = 0; i < 32; i++) printf("%02x", pub[i]);
        printf("\n");
        return 0;
    }
    if (!strcmp(c, "init") && argc >= 5 && argc <= 7) {
        cvs_ledger *l;
        if (hex32(argv[3], seed)) return usage();
        int have_nonce = argc == 7;
        if (have_nonce && hex16(argv[6], nonce)) return usage();
        if (cvs_ledger_init_at(argv[2], seed, argv[4], argc >= 6 ? argv[5] : NULL, have_nonce ? nonce : NULL, &l)) return err();
        cvs_ledger_close(l);
        return 0;
    }
    if (!strcmp(c, "append") && argc >= 6 && argc <= 8) {
        cvs_ledger *l; uint64_t seq;
        if (hex32(argv[3], seed)) return usage();
        int have_nonce = argc == 8;
        if (have_nonce && hex16(argv[7], nonce)) return usage();
        if (cvs_ledger_open(argv[2], seed, &l)) return err();
        if (cvs_ledger_append(l, argv[4], argv[5], argc >= 7 ? argv[6] : NULL, have_nonce ? nonce : NULL, &seq)) { cvs_ledger_close(l); return err(); }
        printf("%llu\n", (unsigned long long)seq);
        cvs_ledger_close(l);
        return 0;
    }
    if (!strcmp(c, "payload") && argc == 5) {
        cvs_ledger *l; char addr[65];
        if (hex32(argv[3], seed)) return usage();
        FILE *f = fopen(argv[4], "rb");
        if (!f) { fprintf(stderr, "cvseal: cannot read %s\n", argv[4]); return 1; }
        size_t n; char *d = slurp(f, &n); fclose(f);
        if (cvs_ledger_open(argv[2], seed, &l)) return err();
        if (cvs_ledger_put_payload(l, (uint8_t *)d, n, addr)) { cvs_ledger_close(l); return err(); }
        printf("%s\n", addr);
        cvs_ledger_close(l);
        return 0;
    }
    if (!strcmp(c, "rotate") && argc >= 5 && argc <= 7) {
        cvs_ledger *l; uint8_t ns[32];
        if (hex32(argv[3], seed) || hex32(argv[4], ns)) return usage();
        int have_nonce = argc == 7;
        if (have_nonce && hex16(argv[6], nonce)) return usage();
        if (cvs_ledger_open(argv[2], seed, &l)) return err();
        if (cvs_ledger_rotate(l, ns, argc >= 6 ? argv[5] : NULL, have_nonce ? nonce : NULL)) { cvs_ledger_close(l); return err(); }
        cvs_ledger_close(l);
        return 0;
    }
    if (!strcmp(c, "verify") && (argc == 4 || argc == 5)) {
        uint64_t n = 0;
        if (cvs_verify_file(argv[2], argv[3], argc == 5 ? argv[4] : NULL, &n)) return err();
        printf("ok: %llu records\n", (unsigned long long)n);
        return 0;
    }
    if (!strcmp(c, "export") && argc == 3) {
        extern int cvs_export_file(const char *path);
        return 0 * argc + cvs_export_file(argv[2]);
    }
    if (!strcmp(c, "reseal") && argc == 3) {
        extern int cvs_reseal_stream(const char *seeds_file, FILE *in, FILE *out);
        return cvs_reseal_stream(argv[2], stdin, stdout);
    }
    if (!strcmp(c, "roots") && argc == 2) {
        extern int cvs_roots_stream(FILE *in, FILE *out);
        return cvs_roots_stream(stdin, stdout);
    }
    return usage();
}
