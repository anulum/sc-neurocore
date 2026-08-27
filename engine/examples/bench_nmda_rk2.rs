// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — NMDA dual-identity Rust benchmark

use sc_neurocore_engine::neurons::{NMDANeuron, SCWBNMDAMagnesiumBlockNeuron};
use std::hint::black_box;
use std::time::Instant;

const STEPS: usize = 20_000;
const REPEATS: usize = 3;

fn source_once() -> (f64, [f64; 5]) {
    let mut state = NMDANeuron::new();
    let started = Instant::now();
    for _ in 0..STEPS {
        state.try_step(black_box(0.6)).expect("valid source step");
    }
    (
        started.elapsed().as_nanos() as f64 / STEPS as f64,
        [
            state.v,
            state.x_nmda,
            state.s_nmda,
            state.ca,
            state.refractory_remaining,
        ],
    )
}

fn sc_once() -> (f64, [f64; 4]) {
    let mut state = SCWBNMDAMagnesiumBlockNeuron::new();
    let started = Instant::now();
    for _ in 0..STEPS {
        state.try_step(black_box(5.0)).expect("valid SC step");
    }
    (
        started.elapsed().as_nanos() as f64 / STEPS as f64,
        [state.v, state.h, state.n, state.s_nmda],
    )
}

fn main() {
    for _ in 0..REPEATS {
        let (ns, state) = source_once();
        println!(
            "source_ns={ns}\nsource_state={},{},{},{},{}",
            state[0], state[1], state[2], state[3], state[4]
        );
        let (ns, state) = sc_once();
        println!(
            "sc_ns={ns}\nsc_state={},{},{},{}",
            state[0], state[1], state[2], state[3]
        );
    }
}
