// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public HDL identifier admission

//! Refuse names that CIRCT silently changes before a backend exposes a different ABI.

use sc_neurocore_engine::ir::{builder::ScGraphBuilder, emit_mlir, emit_sv, graph::ScType, verify};

#[test]
fn predefined_macro_names_cannot_silently_change_modules_or_ports() {
    for component in ["module", "input", "output"] {
        let mut b = ScGraphBuilder::new(if component == "module" {
            "SYNTHESIS"
        } else {
            "SYNTHESIS_MODEL"
        });
        let value = b.input(
            if component == "input" {
                "SYNTHESIS"
            } else {
                "value"
            },
            ScType::Bool,
        );
        b.output(
            if component == "output" {
                "SYNTHESIS"
            } else {
                "result"
            },
            value,
        );
        let graph = b.build();
        let errors = verify::verify(&graph).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.message.contains("reserved by CIRCT")));
        for result in [emit_sv::emit(&graph), emit_mlir::emit(&graph)] {
            assert!(result.unwrap_err().contains("reserved by CIRCT"));
        }
    }
}
