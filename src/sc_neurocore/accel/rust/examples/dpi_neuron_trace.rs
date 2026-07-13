// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Executable Rust-safety DPI trace probe

use sc_neurocore_safety::dpi_neuron::DPINeuron;
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
        21,
        "usage: dpi_neuron_trace I_MEM I_AHP REFRACTORY_TIME I_THRESHOLD \
         I_RESET I_REST I_TAU I_G I_TAU_AHP I_GA I_SPIKE I_0 KAPPA ALPHA \
         TAU TAU_AHP REFRACTORY_PERIOD DT STEPS CURRENT"
    );
    let mut neuron = DPINeuron {
        i_mem: parse(&args[1], "i_mem"),
        i_ahp: parse(&args[2], "i_ahp"),
        refractory_time: parse(&args[3], "refractory_time"),
        i_threshold: parse(&args[4], "i_threshold"),
        i_reset: parse(&args[5], "i_reset"),
        i_rest: parse(&args[6], "i_rest"),
        i_tau: parse(&args[7], "i_tau"),
        i_g: parse(&args[8], "i_g"),
        i_tau_ahp: parse(&args[9], "i_tau_ahp"),
        i_ga: parse(&args[10], "i_ga"),
        i_spike: parse(&args[11], "i_spike"),
        i_0: parse(&args[12], "i_0"),
        kappa: parse(&args[13], "kappa"),
        alpha: parse(&args[14], "alpha"),
        tau: parse(&args[15], "tau"),
        tau_ahp: parse(&args[16], "tau_ahp"),
        refractory_period: parse(&args[17], "refractory_period"),
        dt: parse(&args[18], "dt"),
    };
    let steps: usize = parse(&args[19], "steps");
    let current: f64 = parse(&args[20], "current");
    for _ in 0..steps {
        let spike = neuron
            .step(current)
            .unwrap_or_else(|message| panic!("DPI step rejected: {message}"));
        println!(
            "DPI_TRACE {spike} {:.17} {:.17} {:.17}",
            neuron.i_mem, neuron.i_ahp, neuron.refractory_time
        );
    }
}
