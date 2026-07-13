// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Executable Rust-safety Quadratic IF trace probe

use sc_neurocore_safety::quadratic_if::QuadraticIFNeuron;
use std::env;

fn parse<T: std::str::FromStr>(value: &str, name: &str) -> T {
    value
        .parse::<T>()
        .unwrap_or_else(|_| panic!("invalid {name}: {value}"))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(
        args.len(),
        7,
        "usage: quadratic_if_trace V V_RESET V_PEAK DT STEPS CURRENT"
    );
    let mut neuron = QuadraticIFNeuron {
        v: parse(&args[1], "v"),
        v_reset: parse(&args[2], "v_reset"),
        v_peak: parse(&args[3], "v_peak"),
        dt: parse(&args[4], "dt"),
    };
    let steps: usize = parse(&args[5], "steps");
    let current: f64 = parse(&args[6], "current");
    for _ in 0..steps {
        let spike = neuron
            .step(current)
            .unwrap_or_else(|message| panic!("quadratic-if step rejected: {message}"));
        println!("QIF_TRACE {spike} {:.17}", neuron.v);
    }
}
