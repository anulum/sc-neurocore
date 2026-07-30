// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! JSON command adapter for public Rust SC Compte network dispatch.

use sc_neurocore_engine::sc_compte_wm_network::{
    SCCompteWMNetwork, SCCompteWMNetworkSpec, SCCompteWMRunReceipt, SCCompteWMStimulus,
    SCCompteWMStimulusKind,
};
use std::env;
use std::time::Instant;

struct Options {
    duration_ms: f64,
    statistics_window_ms: f64,
    spec: SCCompteWMNetworkSpec,
    stimuli: Vec<SCCompteWMStimulus>,
}

fn parse_stimulus(value: &str) -> Result<SCCompteWMStimulus, String> {
    let fields: Vec<&str> = value.split(',').collect();
    if fields.len() != 5 {
        return Err("stimulus must have start,duration,current,kind,center".into());
    }
    let kind = match fields[3] {
        "localized_cue" => SCCompteWMStimulusKind::LocalizedCue,
        "global_current" => SCCompteWMStimulusKind::GlobalCurrent,
        _ => return Err("invalid stimulus kind".into()),
    };
    let center_deg = if fields[4] == "none" {
        None
    } else {
        Some(fields[4].parse().map_err(|_| "invalid stimulus center")?)
    };
    Ok(SCCompteWMStimulus {
        start_ms: fields[0].parse().map_err(|_| "invalid stimulus start")?,
        duration_ms: fields[1].parse().map_err(|_| "invalid stimulus duration")?,
        current_pa: fields[2].parse().map_err(|_| "invalid stimulus current")?,
        kind,
        center_deg,
    })
}

fn parse_options() -> Result<Options, String> {
    let mut duration_ms = None;
    let mut statistics_window_ms = None;
    let mut spec = SCCompteWMNetworkSpec::default();
    let mut stimuli = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--duration-ms" => {
                duration_ms = Some(
                    args.next()
                        .ok_or("missing duration")?
                        .parse()
                        .map_err(|_| "invalid duration")?,
                );
            }
            "--statistics-window-ms" => {
                statistics_window_ms = Some(
                    args.next()
                        .ok_or("missing statistics window")?
                        .parse()
                        .map_err(|_| "invalid statistics window")?,
                );
            }
            "--seed" => {
                spec.seed = args
                    .next()
                    .ok_or("missing seed")?
                    .parse()
                    .map_err(|_| "invalid seed")?;
            }
            "--structured-ei" => spec.structured_ei = true,
            "--modulated" => spec.modulated = true,
            "--allow-recurrent-autapses" => spec.allow_recurrent_autapses = true,
            "--stimulus" => stimuli.push(parse_stimulus(
                &args.next().ok_or("missing stimulus encoding")?,
            )?),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    Ok(Options {
        duration_ms: duration_ms.ok_or("--duration-ms is required")?,
        statistics_window_ms: statistics_window_ms.ok_or("--statistics-window-ms is required")?,
        spec,
        stimuli,
    })
}

fn print_receipt(receipt: &SCCompteWMRunReceipt, execution_ns: u128) {
    print!(
        "{{\"runtime\":\"rust\",\"execution_ns\":{execution_ns},\"specification_version\":\"{}\",\"seed\":{},\"duration_ms\":{},\"steps\":{},\"excitatory_spikes\":{},\"inhibitory_spikes\":{},\"input_sha256\":\"{}\",\"spike_sha256\":\"{}\",\"final_state_sha256\":\"{}\",\"windows\":[",
        receipt.specification_version,
        receipt.seed,
        receipt.duration_ms,
        receipt.steps,
        receipt.excitatory_spikes,
        receipt.inhibitory_spikes,
        receipt.input_sha256,
        receipt.spike_sha256,
        receipt.final_state_sha256,
    );
    for (index, window) in receipt.windows.iter().enumerate() {
        if index > 0 {
            print!(",");
        }
        print!(
            "{{\"start_ms\":{},\"end_ms\":{},\"excitatory_spikes\":{},\"inhibitory_spikes\":{},\"statistics\":",
            window.start_ms, window.end_ms, window.excitatory_spikes, window.inhibitory_spikes
        );
        if let Some(statistics) = window.statistics {
            print!(
                "{{\"excitatory_rate_hz\":{},\"inhibitory_rate_hz\":{},\"bump_angle_deg\":{},\"resultant_length\":{},\"circular_width_deg\":",
                statistics.excitatory_rate_hz,
                statistics.inhibitory_rate_hz,
                statistics.bump_angle_deg,
                statistics.resultant_length,
            );
            if let Some(width) = statistics.circular_width_deg {
                print!("{width}");
            } else {
                print!("null");
            }
            print!("}}");
        } else {
            print!("null");
        }
        print!("}}");
    }
    println!("]}}");
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    let mut network = SCCompteWMNetwork::new(options.spec, None).map_err(str::to_owned)?;
    let started = Instant::now();
    let receipt = network
        .run(
            options.duration_ms,
            &options.stimuli,
            options.statistics_window_ms,
        )
        .map_err(str::to_owned)?;
    print_receipt(&receipt, started.elapsed().as_nanos());
    Ok(())
}
