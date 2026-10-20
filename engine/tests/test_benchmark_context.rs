// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Measured native benchmark hardware context

//! Check the maintained context reader against actual kernel observations.

#[path = "../examples/benchmark_context.rs"]
mod benchmark_context;
use std::fs;

fn kernel_field(path: &str, field: &str) -> String {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == field).then(|| value.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unavailable".into())
}

#[test]
fn actual_hardware_capture_does_not_infer_cpu_exclusivity_from_affinity() {
    let context = benchmark_context::measurement_context_json(&benchmark_context::load_average());
    assert!(context.contains("\"runtime_cpuset_shield_claimed\": false"));
    for (key, expected) in [
        ("cpu_model", kernel_field("/proc/cpuinfo", "model name")),
        ("ram_total", kernel_field("/proc/meminfo", "MemTotal")),
        (
            "kernel_release",
            fs::read_to_string("/proc/sys/kernel/osrelease")
                .map(|x| x.trim().to_string())
                .unwrap_or_else(|_| "unavailable".into()),
        ),
    ] {
        assert!(
            context.contains(&format!(
                "\"{key}\": {}",
                benchmark_context::json_string(&expected)
            )),
            "{context}"
        );
    }
    assert!(benchmark_context::rust_version().starts_with("rustc "));
}
