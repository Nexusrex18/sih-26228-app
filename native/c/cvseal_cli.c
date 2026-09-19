/* cvseal — command line for the C core (gate C9). Used by the differential tests and by operators without Python. */
#include "cvseal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(void) {
    fputs("usage:\n"
          "  cvseal canon                              JSON on stdin -> canonical bytes on stdout\n"
          "  cvseal keyid SEEDHEX                      print the key id and public key\n"
          "  cvseal init DB SEEDHEX MANIFEST_JSON [TIME [NONCEHEX]]   create a ledger (genesis)\n"
          "  cvseal append DB SEEDHEX TYPE BODY_JSON [TIME [NONCEHEX]]\n"
          "  cvseal payload DB SEEDHEX FILE            store a payload, print its address\n"
          "  cvseal rotate DB SEEDHEX NEW_SEEDHEX [TIME [NONCEHEX]]\n"
          "  cvseal verify FILE PUBKEYHEX [MANIFEST_HASH]   verify a ledger (SQLite or JSONL export)\n"
          "  cvseal export DB                          print the strict JSONL export\n"
          "  cvseal reseal SEEDS_FILE                  re-sign and re-chain a JSONL export from stdin (byte-identity check)\n"
          "  cvseal roots                              JSONL export on stdin -> Merkle root of every prefix\n",
          stderr);
    return 64;
}
static int hex32(const char *h, uint8_t out[32]) {
    if (strlen(h) != 64) return 1;
    for (int i = 0; i < 32; i++) { unsigned v; if (sscanf(h + 2 * i, "%2x", &v) != 1) return 1; out[i] = (uint8_t)v; }
    return 0;
}
static int hex16(const char *h, uint8_t out[16]) {
    if (strlen(h) != 32) return 1;
    for (int i = 0; i < 16; i++) { unsigned v; if (sscanf(h + 2 * i, "%2x", &v) != 1) return 1; out[i] = (uint8_t)v; }
    return 0;
}
static char *slurp(FILE *f, size_t *n) {
    size_t cap = 1 << 16, len = 0;
    char *b = malloc(cap);
    size_t k;
    while ((k = fread(b + len, 1, cap - len, f)) > 0) { len += k; if (len == cap) { cap *= 2; b = realloc(b, cap); } }
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
