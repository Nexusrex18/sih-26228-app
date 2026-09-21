use cvseal::{canonicalise, verify_file, Ledger, Seed};

fn seed(base: u8) -> Seed {
    let mut s = [0u8; 32];
    for (i, b) in s.iter_mut().enumerate() {
        *b = base + i as u8;
    }
    s
}
fn h(c: char, n: usize) -> String {
    c.to_string().repeat(n)
}
fn model() -> String {
    format!(
        r#"{{"model":{{"id":"m","weights_sha256":"{}","format":"onnx","arch_hash":"{}"}},"config":{{"preprocess_hash":"{}","preprocess_ref":"sha256:{}","postprocess_hash":"{}","runtime":"rt","version_pins_hash":"{}","code_commit":"{}"}}}}"#,
        h('a', 64), h('b', 64), h('c', 64), h('c', 64), h('d', 64), h('e', 64), h('0', 40)
    )
}
fn inference() -> String {
    let m = model();
    format!(
        r#"{{"input":{{"sha256":"{}","source_kind":"encoded_file","phash":null,"phash_omitted_reason":"not_computed","dims":[8,8]}},{},"output":{{"jcs_sha256":"{}","decision_sha256":"{}","raw_jcs_sha256":"{}","payload_ref":"sha256:{}"}}}}"#,
        h('1', 64), &m[1..m.len() - 1], h('2', 64), h('3', 64), h('4', 64), h('2', 64)
    )
}

#[test]
fn canonicalisation_matches_the_spec_and_refuses_floats() {
    assert_eq!(canonicalise(r#" { "b" : [1,2], "a" : "xA" } "#).unwrap(), br#"{"a":"xA","b":[1,2]}"#);
    assert!(canonicalise(r#"{"a":1.5}"#).unwrap_err().0.contains("float"));
    assert!(canonicalise(r#"{"a":"café"}"#).is_err());
}

#[test]
fn a_ledger_written_from_rust_verifies_and_survives_rotation() {
    let dir = std::env::var("CVSEAL_OUT").unwrap_or_else(|_| std::env::temp_dir().join(format!("cvseal-rs-{}", std::process::id())).to_string_lossy().into_owned());
    std::fs::create_dir_all(&dir).unwrap();
    let path = format!("{}/rust.db", dir);
    let _ = std::fs::remove_file(&path);
    let (a, b) = (seed(0), seed(100));
    {
        let mut led = Ledger::init(&path, &a, &format!(r#"{{"device_id":"rs-01","unit":"u","profile_hash":"{}","checkpoint_every":5}}"#, h('7', 64))).unwrap();
        assert_eq!(led.append("model_registration", &model()).unwrap(), 1);
        for _ in 0..8 {
            led.append("inference", &inference()).unwrap();
        }
        led.rotate(&b).unwrap();
        for _ in 0..4 {
            led.append("inference", &inference()).unwrap();
        }
        let addr = led.put_payload(br#"{"task":"classify"}"#).unwrap();
        assert_eq!(addr.len(), 64);
        assert!(led.size() > 14);
    }
    // the outgoing key can no longer open the ledger; the incoming one can
    assert!(Ledger::open(&path, &a).unwrap_err().0.contains("active signing key"));
    Ledger::open(&path, &b).unwrap();
    // public key of seed(0) is the same the Python tests use; the verifier is given it as the trust root would
    let pubkey = std::env::var("CVSEAL_PUBKEY").unwrap_or_else(|_| "".into());
    if !pubkey.is_empty() {
        let n = verify_file(&path, &pubkey, None).unwrap();
        assert!(n > 14);
        assert!(verify_file(&path, &pubkey, Some(&h('0', 64))).unwrap_err().0.contains("different deployment"));
    }
}
