// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Executable Rust-safety Theta trace probe

use sc_neurocore_safety::theta::ThetaNeuron;
use std::env;

fn parse<T: std::str::FromStr>(value: &str, name: &str) -> T {
    value
        .parse::<T>()
        .unwrap_or_else(|_| panic!("invalid {name}: {value}"))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 5, "usage: theta_trace THETA DT STEPS CURRENT");
    let mut neuron = ThetaNeuron {
        theta: parse(&args[1], "theta"),
        dt: parse(&args[2], "dt"),
    };
    let steps: usize = parse(&args[3], "steps");
    let current: f64 = parse(&args[4], "current");
    for _ in 0..steps {
        let spike = neuron
            .step(current)
            .unwrap_or_else(|message| panic!("theta step rejected: {message}"));
        println!("THETA_TRACE {spike} {:.17}", neuron.theta);
    }
}
