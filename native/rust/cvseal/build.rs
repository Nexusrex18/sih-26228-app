// Compiles ../../c/cvseal.c with the system C compiler (no `cc` crate: this crate has no dependencies) and links
// libsodium and sqlite3.
use std::{env, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let c_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../c");
    let obj = out.join("cvseal.o");
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let st = Command::new(&cc)
        .args(["-std=c11", "-O2", "-fPIC", "-c"])
        .arg(c_dir.join("cvseal.c"))
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to run the C compiler");
    assert!(st.success(), "compiling cvseal.c failed");
    let lib = out.join("libcvseal_c.a");
    let st = Command::new("ar").args(["rcs"]).arg(&lib).arg(&obj).status().expect("failed to run ar");
    assert!(st.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=cvseal_c");
    println!("cargo:rustc-link-lib=sodium");
    println!("cargo:rustc-link-lib=sqlite3");
    println!("cargo:rerun-if-changed={}", c_dir.join("cvseal.c").display());
    println!("cargo:rerun-if-changed={}", c_dir.join("cvseal.h").display());
}
