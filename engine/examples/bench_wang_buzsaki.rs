// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wang-Buzsaki source benchmark

use sc_neurocore_engine::neurons::WangBuzsakiNeuron;
use std::hint::black_box;
use std::time::Instant;

const STEPS: usize = 20_000;
const REPEATS: usize = 3;
const CURRENT: f64 = 10.0;

fn main() {
    for _ in 0..REPEATS {
        let mut neuron = WangBuzsakiNeuron::new();
        let mut spikes = 0_i32;
        let started = Instant::now();
        for _ in 0..STEPS {
            spikes += neuron.step(black_box(CURRENT));
        }
        let ns = started.elapsed().as_nanos() as f64 / STEPS as f64;
        println!("ns={ns}");
        println!("spikes={spikes}");
        println!("state={},{},{}", neuron.v, neuron.h, neuron.n);
    }
}
