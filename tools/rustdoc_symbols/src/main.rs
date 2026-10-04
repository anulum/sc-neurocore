// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — native Rust declaration source and identity protocol

//! Parse exact Rust sources into qualified declaration identities and byte spans.
#![deny(missing_docs)]

mod items;

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

/// Parsed declarations and the hash of the exact source submitted to syn.
#[derive(Serialize)]
struct Source {
    /// SHA-256 of the byte-preserving UTF-8 parser input.
    source_sha256: String,
    /// Syntax-qualified symbol identities with identifier byte ranges.
    symbols: Vec<items::Symbol>,
}

/// Versioned native source and declaration response.
#[derive(Serialize)]
struct Report {
    /// Protocol identifier for source hashes and byte span interpretation.
    schema_version: &'static str,
    /// Exact repository-relative source paths, ordered for reproducibility.
    sources: BTreeMap<String, Source>,
}

/// Require regular, unique, normalized Rust source paths inside the actual root.
fn source_path(
    root: &Path,
    raw: &str,
    seen: &mut HashSet<String>,
) -> Result<PathBuf, Box<dyn Error>> {
    let path = Path::new(raw);
    if raw.is_empty()
        || raw.trim() != raw
        || raw.chars().any(char::is_control)
        || raw.contains('\\')
        || !raw.ends_with(".rs")
        || !path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
        || path.components().collect::<PathBuf>().as_os_str() != path.as_os_str()
        || !seen.insert(raw.to_owned())
    {
        return Err("Rust source paths must be unique normalized relative paths".into());
    }
    let absolute = root.join(path);
    if absolute.canonicalize()? != absolute || !absolute.is_file() {
        return Err("Rust source inputs must be regular files without symbolic links".into());
    }
    Ok(absolute)
}

/// Execute the native parser and publish only a complete source cohort.
fn run() -> Result<(), Box<dyn Error>> {
    let paths: Vec<String> = serde_json::from_reader(io::stdin().lock())?;
    if paths.is_empty() {
        return Err("At least one Rust source is required".into());
    }
    let root = std::env::current_dir()?.canonicalize()?;
    let mut seen = HashSet::new();
    let mut sources = BTreeMap::new();
    for raw in paths {
        let path = source_path(&root, &raw, &mut seen)?;
        let source = fs::read_to_string(path)?;
        let ast = syn::parse_file(&source)?;
        let mut source_sha256 = String::with_capacity(64);
        for byte in Sha256::digest(source.as_bytes()) {
            write!(source_sha256, "{byte:02x}")?;
        }
        sources.insert(
            raw,
            Source {
                source_sha256,
                symbols: items::collect(&ast.items, ""),
            },
        );
    }
    let mut output = io::stdout().lock();
    serde_json::to_writer(
        &mut output,
        &Report {
            schema_version: "sc-neurocore.rustdoc-symbols.v1",
            sources,
        },
    )?;
    std::io::Write::flush(&mut output)?;
    Ok(())
}

/// Report parser/input/output failures without publishing partial identities.
fn main() {
    if let Err(error) = run() {
        eprintln!("rustdoc_symbols: {error}");
        std::process::exit(2);
    }
}
