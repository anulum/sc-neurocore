// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Larter-Breakspear dual-identity Rust benchmark

use sc_neurocore_engine::neurons::{LarterBreakspearNeuron, SCDecoupledAdaptationIonMassNeuron};
use std::hint::black_box;
use std::time::Instant;

const STEPS: usize = 20_000;
const REPEATS: usize = 3;

fn source_once() -> (f64, [f64; 3]) {
    let mut state = LarterBreakspearNeuron::new();
    let started = Instant::now();
    for _ in 0..STEPS {
        state.try_step(black_box(0.0)).expect("valid source step");
    }
    (
        started.elapsed().as_nanos() as f64 / STEPS as f64,
        [state.v, state.w, state.z],
    )
}

fn sc_once() -> (f64, [f64; 3]) {
    let mut state = SCDecoupledAdaptationIonMassNeuron::new();
    let started = Instant::now();
    for _ in 0..STEPS {
        state.try_step(black_box(0.0)).expect("valid SC step");
    }
    (
        started.elapsed().as_nanos() as f64 / STEPS as f64,
        [state.v, state.w, state.z],
    )
}

fn main() {
    for _ in 0..REPEATS {
        let (ns, state) = source_once();
        println!(
            "source_ns={ns}\nsource_state={},{},{}",
            state[0], state[1], state[2]
        );
        let (ns, state) = sc_once();
        println!(
            "sc_ns={ns}\nsc_state={},{},{}",
            state[0], state[1], state[2]
        );
    }
}
