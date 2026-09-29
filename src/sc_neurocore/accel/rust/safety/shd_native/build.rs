// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — System HDF5 linker metadata

//! Locate the operator-installed HDF5 library without resolving Cargo dependencies.

use std::{error::Error, process::Command};

fn main() -> Result<(), Box<dyn Error>> {
    let output = Command::new("pkg-config")
        .args(["--libs", "hdf5"])
        .output()?;
    if !output.status.success() {
        return Err("system HDF5 pkg-config metadata is unavailable".into());
    }
    for flag in std::str::from_utf8(&output.stdout)?.split_whitespace() {
        if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(name) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={name}");
        } else {
            return Err(format!("unsupported HDF5 linker flag: {flag}").into());
        }
    }
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR");
    Ok(())
}
