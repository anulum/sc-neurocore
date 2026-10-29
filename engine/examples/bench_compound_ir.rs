// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source-bound public compound IR compilation benchmark

//! Time public parse, verification and both emitters against retained compiler outputs.

mod benchmark_context;

use benchmark_context::{json_string, load_average, measurement_context_json, rust_version};
use sc_neurocore_engine::ir::{emit_mlir, emit_sv, parser, printer, verify};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    hint::black_box,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Compile the complete public graph without retaining any hidden prepared state.
fn compile(text: &str) -> Result<(String, String), Box<dyn Error>> {
    let graph = parser::parse(text).map_err(|error| error.to_string())?;
    verify::verify(&graph).map_err(|errors| format!("{errors:?}"))?;
    if printer::print(&graph) != text {
        return Err("benchmark graph text must be canonical".into());
    }
    Ok((emit_sv::emit(&graph)?, emit_mlir::emit(&graph)?))
}

/// Hash actual benchmark inputs and expected outputs rather than accepting an identity label.
fn sha256(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in &Sha256::digest(bytes) {
        text.push(HEX[(byte >> 4) as usize] as char);
        text.push(HEX[(byte & 15) as usize] as char);
    }
    text
}

/// Refuse invalid workload sizes before reading inputs or timing any compiler work.
fn positive(value: Option<&String>, default: usize) -> Result<usize, Box<dyn Error>> {
    let number = value.map(|s| s.parse()).transpose()?.unwrap_or(default);
    if number == 0 {
        return Err("iterations and repeats must be positive".into());
    }
    Ok(number)
}

/// Qualify both outputs once, then retain all repeated per-graph compilation durations.
fn measure(case: &Path, iterations: usize, repeats: usize) -> Result<String, Box<dyn Error>> {
    let text = fs::read_to_string(case.join("graph.scir"))?;
    let sv = fs::read_to_string(case.join("graph.sv"))?;
    let mlir = fs::read_to_string(case.join("graph.mlir"))?;
    let actual = compile(&text)?;
    if actual.0 != sv || actual.1 != mlir {
        return Err(format!(
            "compiler output differs from the qualified corpus: {}",
            case.display()
        )
        .into());
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(compile(black_box(&text))?);
        }
        samples.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let raw = samples
        .iter()
        .map(|n| format!("{n:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    let name = case
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("non-UTF-8 corpus name")?;
    Ok(format!(
        "{}:{{\"source_sha256\":{},\"sv_sha256\":{},\"mlir_sha256\":{},\"sv_bytes\":{},\"mlir_bytes\":{},\"median_ns\":{:.3},\"samples_ns\":[{}]}}",
        json_string(name), json_string(&sha256(text.as_bytes())), json_string(&sha256(sv.as_bytes())),
        json_string(&sha256(mlir.as_bytes())), sv.len(), mlir.len(), sorted[sorted.len()/2], raw,
    ))
}

/// Run the registered benchmark CLI on a nonempty corpus from the real integration tests.
pub fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(2..=4).contains(&args.len()) {
        return Err("usage: bench_compound_ir CORPUS [ITERATIONS=10] [REPEATS=7]".into());
    }
    let iterations = positive(args.get(2), 10)?;
    let repeats = positive(args.get(3), 7)?;
    let before = load_average();
    let mut cases = fs::read_dir(&args[1])?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()?;
    cases.retain(|path| path.is_dir());
    cases.sort();
    if cases.is_empty() {
        return Err("benchmark corpus has no circuit directories".into());
    }
    let rows = cases
        .iter()
        .map(|case| measure(case, iterations, repeats))
        .collect::<Result<Vec<_>, _>>()?;
    let command = args
        .iter()
        .map(|arg| json_string(arg))
        .collect::<Vec<_>>()
        .join(",");
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    println!(
        "{{\"schema_version\":1,\"benchmark\":\"compound_ir_compile\",\"language\":\"Rust\",\"performance_promotion_claimed\":false,\"cross_language_comparison\":false,\"hardware_measurement_claimed\":false,\"timestamp_unix\":{},\"iterations\":{},\"repeats\":{},\"debug_assertions\":{},\"build_scope\":{},\"rustc\":{},\"command\":[{}],\"measurement_context\":{},\"case_count\":{},\"cases\":{{{}}}}}",
        timestamp, iterations, repeats, cfg!(debug_assertions),
        json_string(option_env!("SCN_BENCH_BUILD_SCOPE").unwrap_or("Cargo example; see command and debug_assertions")),
        json_string(&rust_version()), command, measurement_context_json(&before), rows.len(), rows.join(","),
    );
    Ok(())
}
