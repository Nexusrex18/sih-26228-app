// cvseal.hpp — C++17 binding over the C core (gate C9). Header-only; link with libcvseal.a -lsodium -lsqlite3.
// Errors are exceptions (cvseal::Error carrying cvs_errmsg()); handles are RAII and move-only.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "../c/cvseal.h"
}

namespace cvseal {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string &what) : std::runtime_error(what) {}
};

inline void check(int rc) {
    if (rc != CVS_OK) throw Error(cvs_errmsg());
}
inline void init() { check(cvs_init()); }

using Seed = std::array<std::uint8_t, 32>;
using Nonce = std::array<std::uint8_t, 16>;

// Canonical bytes (spec section 2) of any JSON text in the profile; throws Error if it is outside it.
inline std::string canonicalise(const std::string &json) {
    init();
    char *out = nullptr;
    std::size_t n = 0;
    check(cvs_canonicalise(json.data(), json.size(), &out, &n));
    std::unique_ptr<char, decltype(&cvs_free)> guard(out, &cvs_free);
    return std::string(out, n);
}

// Verify a ledger file (SQLite store or JSONL export) given the ledger key's public key (64 hex).
// Returns the number of records; throws Error naming the first failure.
inline std::uint64_t verify_file(const std::string &path, const std::string &pubkey_hex,
                                 const std::optional<std::string> &manifest_hash = std::nullopt) {
    init();
    std::uint64_t n = 0;
    check(cvs_verify_file(path.c_str(), pubkey_hex.c_str(), manifest_hash ? manifest_hash->c_str() : nullptr, &n));
    return n;
}

class Ledger {
public:
    static Ledger init(const std::string &path, const Seed &seed, const std::string &manifest_json) {
        cvseal::init();
        cvs_ledger *l = nullptr;
        check(cvs_ledger_init(path.c_str(), seed.data(), manifest_json.c_str(), &l));
        return Ledger(l);
    }
    static Ledger open(const std::string &path, const Seed &seed) {
        cvseal::init();
        cvs_ledger *l = nullptr;
        check(cvs_ledger_open(path.c_str(), seed.data(), &l));
        return Ledger(l);
    }
    Ledger(Ledger &&o) noexcept : l_(o.l_) { o.l_ = nullptr; }
    Ledger &operator=(Ledger &&o) noexcept {
        if (this != &o) { close(); l_ = o.l_; o.l_ = nullptr; }
        return *this;
    }
    Ledger(const Ledger &) = delete;
    Ledger &operator=(const Ledger &) = delete;
    ~Ledger() { close(); }

    // Appends a typed record; `body_json` holds exactly the type's sections. Returns its seq.
    std::uint64_t append(const std::string &type, const std::string &body_json) {
        std::uint64_t seq = 0;
        check(cvs_ledger_append(l_, type.c_str(), body_json.c_str(), nullptr, nullptr, &seq));
        return seq;
    }
    std::string put_payload(const std::string &bytes) {
        char addr[65];
        check(cvs_ledger_put_payload(l_, reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size(), addr));
        return addr;
    }
    void rotate(const Seed &new_seed) { check(cvs_ledger_rotate(l_, new_seed.data(), nullptr, nullptr)); }
    std::uint64_t size() const { return cvs_ledger_size(l_); }

private:
    explicit Ledger(cvs_ledger *l) : l_(l) {}
    void close() { if (l_) { cvs_ledger_close(l_); l_ = nullptr; } }
    cvs_ledger *l_;
};

}  // namespace cvseal
