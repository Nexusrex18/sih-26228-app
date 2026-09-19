// Test for the C++ binding. Writes a ledger (path from argv[1]) that the Python test then verifies.
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "cvseal.hpp"

#define EXPECT(c) do { if (!(c)) { std::fprintf(stderr, "FAILED: %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

int main(int argc, char **argv) {
    if (argc < 3) { std::fputs("usage: test_cvseal LEDGER PUBKEY_HEX\n", stderr); return 64; }
    // canonicalisation: order, whitespace, escapes
    EXPECT(cvseal::canonicalise(" { \"b\" : [1,2] , \"a\" : \"x\\u0041\" } ") == "{\"a\":\"xA\",\"b\":[1,2]}");
    bool refused = false;
    try { cvseal::canonicalise("{\"a\":1.5}"); } catch (const cvseal::Error &e) { refused = true; }
    EXPECT(refused);

    cvseal::Seed seed{};
    for (int i = 0; i < 32; i++) seed[i] = static_cast<std::uint8_t>(i);        // the same key as the Python tests' SEED_A
    cvseal::Seed next{};
    for (int i = 0; i < 32; i++) next[i] = static_cast<std::uint8_t>(100 + i);   // SEED_B
    {
        auto led = cvseal::Ledger::init(argv[1], seed,
            "{\"device_id\":\"cpp-01\",\"unit\":\"u\",\"profile_hash\":\"" + std::string(64, '7') + "\",\"checkpoint_every\":5}");
        EXPECT(led.size() == 1);
        std::string model = "{\"model\":{\"id\":\"m\",\"weights_sha256\":\"" + std::string(64, 'a') + "\",\"format\":\"onnx\",\"arch_hash\":\""
            + std::string(64, 'b') + "\"},\"config\":{\"preprocess_hash\":\"" + std::string(64, 'c') + "\",\"preprocess_ref\":\"sha256:"
            + std::string(64, 'c') + "\",\"postprocess_hash\":\"" + std::string(64, 'd') + "\",\"runtime\":\"rt\",\"version_pins_hash\":\""
            + std::string(64, 'e') + "\",\"code_commit\":\"" + std::string(40, '0') + "\"}}";
        EXPECT(led.append("model_registration", model) == 1);
        std::string inf = "{\"input\":{\"sha256\":\"" + std::string(64, '1') + "\",\"source_kind\":\"encoded_file\",\"phash\":null,"
            "\"phash_omitted_reason\":\"not_computed\",\"dims\":[8,8]},"
            + model.substr(1, model.size() - 2) + ",\"output\":{\"jcs_sha256\":\"" + std::string(64, '2') + "\",\"decision_sha256\":\""
            + std::string(64, '3') + "\",\"raw_jcs_sha256\":\"" + std::string(64, '4') + "\",\"payload_ref\":\"sha256:" + std::string(64, '2') + "\"}}";
        for (int i = 0; i < 7; i++) led.append("inference", inf);
        led.rotate(next);
        auto moved = std::move(led);                                             // RAII + move
        for (int i = 0; i < 3; i++) moved.append("inference", inf);
        EXPECT(moved.size() > 12);
    }
    const std::uint64_t n = cvseal::verify_file(argv[1], argv[2]);
    EXPECT(n > 12);
    bool wrong_key = false;
    try { auto l = cvseal::Ledger::open(argv[1], seed); } catch (const cvseal::Error &) { wrong_key = true; }   // rotated away
    EXPECT(wrong_key);
    std::cout << "cpp ok: " << n << " records\n";
    return 0;
}
