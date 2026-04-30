// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

#![no_main]

use libfuzzer_sys::fuzz_target;
use sc_neurocore_engine::ir::{parser, printer, verify};

fuzz_target!(|data: &[u8]| {
    if data.len() > 16 * 1024 {
        return;
    }
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    if let Ok(graph) = parser::parse(text) {
        let _ = verify::verify(&graph);
        let printed = printer::print(&graph);
        let _ = parser::parse(&printed);
    }
});
