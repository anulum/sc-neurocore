// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

//! Compile the host long-double conversion without Cargo dependencies.
use std::{env, error::Error, path::PathBuf, process::Command};
fn main() -> Result<(), Box<dyn Error>> {
    let output = PathBuf::from(env::var("OUT_DIR")?);
    let object = output.join("extended.o");
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(compiler)
        .args([
            "-std=c11",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            "src/extended.c",
            "-o",
        ])
        .arg(&object)
        .status()?;
    if !status.success() {
        return Err("native extended conversion compilation failed".into());
    }
    let status = Command::new("ar")
        .arg("rcs")
        .arg(output.join("libdvs_extended.a"))
        .arg(object)
        .status()?;
    if !status.success() {
        return Err("native extended conversion archive failed".into());
    }
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=dvs_extended");
    println!("cargo:rerun-if-changed=src/extended.c");
    println!("cargo:rerun-if-env-changed=CC");
    Ok(())
}
