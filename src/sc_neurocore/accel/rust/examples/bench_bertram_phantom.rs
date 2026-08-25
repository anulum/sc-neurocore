// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bertram phantom source-kernel benchmark executable

use std::hint::black_box;
use std::time::Instant;

use sc_neurocore_safety::bertram_phantom::BertramPhantomBurster;

const STEPS: usize = 10_000;

fn main() {
    let mut model = BertramPhantomBurster::new();
    let started = Instant::now();
    let mut events = 0_i32;
    for _ in 0..STEPS {
        events += model.step(black_box(0.0)).expect("enrolled drive is valid");
    }
    let elapsed = started.elapsed().as_nanos();
    println!(
        "{elapsed} {} {} {} {} {events}",
        model.v, model.n, model.s1, model.s2
    );
}
