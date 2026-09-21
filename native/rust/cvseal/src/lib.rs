//! Rust binding over the C core of the inference provenance seal (`cva-seal/1`, gate C9).
//!
//! A safe wrapper over `cvseal.h`: canonical JSON, an append-only signed ledger on the same SQLite store the Python
//! reference reads and writes, key rotation, and chain verification. Errors carry the C core's message.

use std::ffi::{c_char, c_int, CStr, CString};
use std::fmt;
use std::ptr;

#[allow(non_camel_case_types)]
#[repr(C)]
struct cvs_ledger {
    _private: [u8; 0],
}

extern "C" {
    fn cvs_init() -> c_int;
    fn cvs_errmsg() -> *const c_char;
    fn cvs_free(p: *mut std::ffi::c_void);
    fn cvs_canonicalise(inp: *const c_char, n: usize, out: *mut *mut c_char, out_n: *mut usize) -> c_int;
    fn cvs_ledger_init(path: *const c_char, seed: *const u8, manifest: *const c_char, out: *mut *mut cvs_ledger) -> c_int;
    fn cvs_ledger_open(path: *const c_char, seed: *const u8, out: *mut *mut cvs_ledger) -> c_int;
    fn cvs_ledger_close(l: *mut cvs_ledger);
    fn cvs_ledger_append(l: *mut cvs_ledger, ty: *const c_char, body: *const c_char, now: *const c_char, nonce: *const u8, seq: *mut u64) -> c_int;
    fn cvs_ledger_put_payload(l: *mut cvs_ledger, data: *const u8, n: usize, addr: *mut c_char) -> c_int;
    fn cvs_ledger_rotate(l: *mut cvs_ledger, new_seed: *const u8, now: *const c_char, nonce: *const u8) -> c_int;
    fn cvs_ledger_size(l: *mut cvs_ledger) -> u64;
    fn cvs_verify_file(path: *const c_char, pubkey: *const c_char, manifest: *const c_char, n: *mut u64) -> c_int;
}

/// A 32-byte Ed25519 seed.
pub type Seed = [u8; 32];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cvseal: {}", self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

fn last_error() -> Error {
    // SAFETY: cvs_errmsg returns a pointer to a static NUL-terminated buffer.
    Error(unsafe { CStr::from_ptr(cvs_errmsg()) }.to_string_lossy().into_owned())
}
fn check(rc: c_int) -> Result<()> {
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}
fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error("interior NUL byte".into()))
}
fn init() -> Result<()> {
    // SAFETY: idempotent libsodium initialisation.
    check(unsafe { cvs_init() })
}

/// Canonical bytes (spec §2) of any JSON text in the profile.
pub fn canonicalise(json: &str) -> Result<Vec<u8>> {
    init()?;
    let mut out: *mut c_char = ptr::null_mut();
    let mut n = 0usize;
    // SAFETY: pointer/length come from a live &str; out/n are valid for writes.
    check(unsafe { cvs_canonicalise(json.as_ptr() as *const c_char, json.len(), &mut out, &mut n) })?;
    // SAFETY: on success `out` points at n initialised bytes allocated by the C core; we copy then free it.
    let v = unsafe { std::slice::from_raw_parts(out as *const u8, n) }.to_vec();
    unsafe { cvs_free(out as *mut _) };
    Ok(v)
}

/// Verify a ledger file (SQLite store or JSONL export) against the ledger's public key (64 hex).
/// Returns the number of records, or the first failure.
pub fn verify_file(path: &str, pubkey_hex: &str, manifest_hash: Option<&str>) -> Result<u64> {
    init()?;
    let (p, k) = (cstr(path)?, cstr(pubkey_hex)?);
    let m = manifest_hash.map(cstr).transpose()?;
    let mut n = 0u64;
    // SAFETY: all pointers are valid NUL-terminated strings for the duration of the call.
    check(unsafe { cvs_verify_file(p.as_ptr(), k.as_ptr(), m.as_ref().map_or(ptr::null(), |c| c.as_ptr()), &mut n) })?;
    Ok(n)
}

/// An open, writable ledger. Closed on drop.
pub struct Ledger {
    raw: *mut cvs_ledger,
}

impl Ledger {
    pub fn init(path: &str, seed: &Seed, manifest_json: &str) -> Result<Ledger> {
        init()?;
        let (p, m) = (cstr(path)?, cstr(manifest_json)?);
        let mut l: *mut cvs_ledger = ptr::null_mut();
        // SAFETY: valid strings, a 32-byte seed, and a writable out pointer.
        check(unsafe { cvs_ledger_init(p.as_ptr(), seed.as_ptr(), m.as_ptr(), &mut l) })?;
        Ok(Ledger { raw: l })
    }
    pub fn open(path: &str, seed: &Seed) -> Result<Ledger> {
        init()?;
        let p = cstr(path)?;
        let mut l: *mut cvs_ledger = ptr::null_mut();
        // SAFETY: as above.
        check(unsafe { cvs_ledger_open(p.as_ptr(), seed.as_ptr(), &mut l) })?;
        Ok(Ledger { raw: l })
    }
    /// Append a typed record whose `body_json` holds exactly the type's sections. Returns its seq.
    pub fn append(&mut self, record_type: &str, body_json: &str) -> Result<u64> {
        let (t, b) = (cstr(record_type)?, cstr(body_json)?);
        let mut seq = 0u64;
        // SAFETY: `raw` is a live handle owned by self; strings are valid; now/nonce NULL = real clock / CSPRNG.
        check(unsafe { cvs_ledger_append(self.raw, t.as_ptr(), b.as_ptr(), ptr::null(), ptr::null(), &mut seq) })?;
        Ok(seq)
    }
    /// Store a payload; returns its SHA-256 content address (64 hex).
    pub fn put_payload(&mut self, bytes: &[u8]) -> Result<String> {
        let mut addr = [0 as c_char; 65];
        // SAFETY: `addr` has room for 64 hex characters plus NUL.
        check(unsafe { cvs_ledger_put_payload(self.raw, bytes.as_ptr(), bytes.len(), addr.as_mut_ptr()) })?;
        Ok(unsafe { CStr::from_ptr(addr.as_ptr()) }.to_string_lossy().into_owned())
    }
    /// Hand signing to `new_seed`'s key: signed by the outgoing key, with the incoming key's proof of possession.
    pub fn rotate(&mut self, new_seed: &Seed) -> Result<()> {
        // SAFETY: live handle, 32-byte seed, NULL now/nonce.
        check(unsafe { cvs_ledger_rotate(self.raw, new_seed.as_ptr(), ptr::null(), ptr::null()) })
    }
    pub fn size(&self) -> u64 {
        // SAFETY: live handle.
        unsafe { cvs_ledger_size(self.raw) }
    }
}

impl fmt::Debug for Ledger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ledger {{ records: {} }}", self.size()) // never prints key material
    }
}

impl Drop for Ledger {
    fn drop(&mut self) {
        // SAFETY: `raw` was returned by init/open and is closed exactly once.
        unsafe { cvs_ledger_close(self.raw) }
    }
}
