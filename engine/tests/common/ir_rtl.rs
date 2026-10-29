// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Real emitted graph RTL simulation

//! Compile emitted modules with the maintained RTL using the actual Icarus tools.

use std::{
    fs,
    path::Path,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);

/// Simulate a generated top and return its complete public/hierarchical observations.
pub fn simulate(label: &str, sv: &str, bench: &str, primitives: &Path) -> String {
    let dir = std::env::temp_dir().join(format!(
        "sc-ir-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let _scratch = Scratch(dir.clone());
    let top = dir.join("top.sv");
    let tb = dir.join("tb.sv");
    let binary = dir.join("sim.vvp");
    fs::write(&top, sv).unwrap();
    fs::write(&tb, bench).unwrap();
    let mut command = Command::new("iverilog");
    command
        .args(["-g2012", "-Wall", "-s", "tb", "-o"])
        .arg(&binary)
        .arg(&top)
        .arg(&tb);
    for name in [
        "sc_bitstream_encoder.v",
        "sc_bitstream_synapse.v",
        "sc_dotproduct_to_current.v",
        "sc_lif_neuron.v",
        "sc_dense_matrix_layer.v",
        "sc_dcls_layer_core.v",
        "sc_dcls_tent_kernel.v",
        "sc_dcls_axonal_delay.v",
        "sc_graph_forward.v",
        "sc_softmax_attention.v",
        "sc_kuramoto_step.v",
    ] {
        command.arg(primitives.join(name));
    }
    let build = command
        .output()
        .expect("Icarus must be installed for RTL tests");
    assert!(
        build.status.success(),
        "{label}: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let output = Command::new("vvp")
        .arg(&binary)
        .output()
        .expect("vvp must be installed");
    assert!(
        output.status.success(),
        "{label}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    if let Some(root) = std::env::var_os("SCN_RTL_RECEIPT_DIR") {
        let receipt = Path::new(&root).join(label);
        fs::create_dir_all(&receipt).unwrap();
        fs::write(receipt.join("top.sv"), sv).unwrap();
        fs::write(receipt.join("tb.sv"), bench).unwrap();
        fs::write(receipt.join("compile-stderr.log"), &build.stderr).unwrap();
        fs::write(receipt.join("trace.log"), &stdout).unwrap();
    }
    stdout
}
