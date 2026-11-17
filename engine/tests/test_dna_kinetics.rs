// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — DNA kinetic simulation refusal tests

//! The kinetic simulation refuses what it cannot simulate before it runs, and still runs the rest.
//!
//! Before, a gate without its input names indexed past the list and panicked, which the release
//! build turns into an abort; `NAND` and `XOR` ran on the leak alone; a step of 0 asked for an
//! endless allocation; and a gate that wrote `"time"` or another gate's output replaced that trace.

use std::collections::HashMap;

use sc_neurocore_engine::dna::{
    check_kinetics, simulate_kinetics, DnaGateSpec, DnaGateType, KineticConfig, KineticsError,
    MAX_KINETIC_SAMPLES,
};

const ALL_KINDS: [DnaGateType; 9] = [
    DnaGateType::And,
    DnaGateType::Or,
    DnaGateType::Not,
    DnaGateType::Threshold,
    DnaGateType::Mux,
    DnaGateType::Amplifier,
    DnaGateType::Buffer,
    DnaGateType::Nand,
    DnaGateType::Xor,
];

fn gate(kind: DnaGateType, inputs: &[&str], output: &str) -> DnaGateSpec {
    DnaGateSpec {
        gate_type: kind,
        input_names: inputs.iter().map(|name| name.to_string()).collect(),
        output_name: output.to_string(),
        threshold: 0.5,
        leak_rate: 1e-7,
    }
}

/// Run a refused simulation and return its message; the check alone gives the same refusal.
fn refusal(gates: &[DnaGateSpec], duration_s: f64, dt: f64) -> String {
    let inputs = HashMap::from([("A".to_string(), 200.0)]);
    let error = simulate_kinetics(gates, &inputs, duration_s, dt, &KineticConfig::default())
        .expect_err("the run must be refused");
    assert_eq!(
        check_kinetics(gates, &inputs, duration_s, dt),
        Err(error.clone())
    );
    error.to_string()
}

#[test]
fn nand_and_xor_have_no_rate_law_and_are_refused() {
    for kind in [DnaGateType::Nand, DnaGateType::Xor] {
        assert_eq!(
            refusal(&[gate(kind, &["A", "B"], "C")], 10.0, 1.0),
            format!(
                "gate 0 ({}) has no rate law, so it cannot be simulated",
                kind.name()
            ),
        );
    }
}

#[test]
fn a_gate_with_the_wrong_input_name_count_for_its_kind_is_refused() {
    let names = ["A", "B", "C", "D"];
    let cases = [
        (DnaGateType::Not, 0, "exactly 1"),
        (DnaGateType::Not, 2, "exactly 1"),
        (DnaGateType::Threshold, 0, "exactly 1"),
        (DnaGateType::Amplifier, 0, "exactly 1"),
        (DnaGateType::Buffer, 0, "exactly 1"),
        (DnaGateType::Buffer, 2, "exactly 1"),
        (DnaGateType::Mux, 2, "exactly 3"),
        (DnaGateType::Mux, 4, "exactly 3"),
        (DnaGateType::And, 0, "at least 1"),
        (DnaGateType::Or, 0, "at least 1"),
    ];
    for (kind, count, needed) in cases {
        assert_eq!(
            refusal(&[gate(kind, &names[..count], "Y")], 10.0, 1.0),
            format!(
                "gate 0 ({}) has an input name count of {count}; its kind needs {needed}",
                kind.name()
            ),
        );
    }
}

#[test]
fn the_refusal_names_the_first_refused_gate_by_its_position() {
    let gates = [
        gate(DnaGateType::And, &["A"], "X"),
        gate(DnaGateType::Not, &[], "Y"),
        gate(DnaGateType::Nand, &["A", "B"], "Z"),
    ];
    assert_eq!(
        refusal(&gates, 10.0, 1.0),
        "gate 1 (NOT) has an input name count of 0; its kind needs exactly 1"
    );
}

#[test]
fn an_output_that_is_already_a_trace_of_the_result_is_refused() {
    let shared = [
        gate(DnaGateType::And, &["A"], "C"),
        gate(DnaGateType::Or, &["A"], "C"),
    ];
    assert_eq!(
        refusal(&shared, 10.0, 1.0),
        "gate 1's output 'C' is already a trace of the result"
    );
    assert_eq!(
        refusal(&[gate(DnaGateType::Buffer, &["A"], "time")], 10.0, 1.0),
        "gate 0's output 'time' is already a trace of the result"
    );
}

#[test]
fn a_step_that_is_not_finite_and_above_zero_is_refused_before_the_gates() {
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            refusal(&[gate(DnaGateType::Nand, &["A", "B"], "Y")], 10.0, dt),
            "dt must be finite and above 0"
        );
    }
}

#[test]
fn a_duration_that_is_not_finite_or_shorter_than_one_step_is_refused() {
    for duration_s in [f64::NAN, f64::INFINITY, 0.5, -1.0] {
        assert_eq!(
            refusal(&[gate(DnaGateType::Buffer, &["A"], "Y")], duration_s, 1.0),
            "duration_s must be finite and at least dt"
        );
    }
}

#[test]
fn a_run_above_the_sample_limit_is_refused_and_a_run_at_it_is_not() {
    let one = [gate(DnaGateType::Buffer, &["A"], "Y")];
    let steps_at_limit = MAX_KINETIC_SAMPLES / 2;
    assert_eq!(MAX_KINETIC_SAMPLES, 1 << 25);
    assert_eq!(
        check_kinetics(&one, &HashMap::new(), steps_at_limit as f64, 1.0),
        Ok(steps_at_limit)
    );
    assert_eq!(
        check_kinetics(&one, &HashMap::new(), steps_at_limit as f64 + 1.0, 1.0),
        Err(KineticsError::TooManySamples)
    );
    let message = "the run would hold more than 33554432 samples over its traces; \
                   use a larger dt or a shorter duration_s";
    assert_eq!(refusal(&one, 10.0, 1e-300), message);
    // The ratio itself overflows to infinity here.
    assert_eq!(refusal(&one, 1e10, 5e-324), message);
}

#[test]
fn every_kind_name_parses_in_either_case_and_no_other_name_does() {
    for kind in ALL_KINDS {
        assert_eq!(DnaGateType::parse(kind.name()), Some(kind));
        assert_eq!(DnaGateType::parse(&kind.name().to_lowercase()), Some(kind));
    }
    for name in ["", "FOO", "CATALYTIC", "AND "] {
        assert_eq!(DnaGateType::parse(name), None);
    }
}

#[test]
fn an_accepted_run_has_one_sample_per_step_for_every_kind_with_a_rate_law() {
    let gates = [
        gate(DnaGateType::And, &["A", "B"], "and"),
        gate(DnaGateType::Or, &["A", "B"], "or"),
        gate(DnaGateType::Not, &["A"], "not"),
        gate(DnaGateType::Threshold, &["A"], "threshold"),
        gate(DnaGateType::Mux, &["S", "A", "B"], "mux"),
        gate(DnaGateType::Amplifier, &["B"], "amplifier"),
        gate(DnaGateType::Buffer, &["A"], "buffer"),
    ];
    let inputs = HashMap::from([
        ("A".to_string(), 150.0),
        ("B".to_string(), 50.0),
        ("S".to_string(), 100.0),
    ]);
    let config = KineticConfig::default();
    assert_eq!(check_kinetics(&gates, &inputs, 10.0, 0.5), Ok(20));
    let result = simulate_kinetics(&gates, &inputs, 10.0, 0.5, &config).expect("accepted");
    let time: Vec<f64> = (0..20).map(|step| step as f64 * 0.5).collect();
    assert_eq!(result["time"], time);
    assert_eq!(result.len(), gates.len() + 1);
    for spec in &gates {
        let trace = &result[&spec.output_name];
        assert_eq!(trace.len(), 20);
        assert_eq!(trace[0], 0.0);
        assert!(trace.iter().all(|&c| (0.0..=config.max_conc).contains(&c)));
        assert!(trace[19] > 0.0, "{} never rose", spec.output_name);
    }
}

#[test]
fn an_input_concentration_that_is_not_finite_or_is_negative_is_refused() {
    let one = [gate(DnaGateType::Buffer, &["A"], "Y")];
    let config = KineticConfig::default();
    for value in [
        -50.0,
        -f64::MIN_POSITIVE,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let inputs = HashMap::from([("A".to_string(), value)]);
        let error = simulate_kinetics(&one, &inputs, 10.0, 1.0, &config).expect_err("refused");
        assert_eq!(check_kinetics(&one, &inputs, 10.0, 1.0), Err(error.clone()));
        assert_eq!(
            error.to_string(),
            "input 'A' must have a finite concentration of at least 0"
        );
    }
    // Zero is a concentration; so is the largest finite value.
    for value in [0.0, f64::MAX] {
        let inputs = HashMap::from([("A".to_string(), value)]);
        assert_eq!(check_kinetics(&one, &inputs, 10.0, 1.0), Ok(10));
    }
}

#[test]
fn the_first_refused_input_is_named_in_sorted_order_after_the_step_and_before_the_gates() {
    let inputs = HashMap::from([("B".to_string(), -1.0), ("A".to_string(), f64::NAN)]);
    let one = [gate(DnaGateType::Buffer, &["A"], "Y")];
    assert_eq!(
        check_kinetics(&one, &inputs, 10.0, 1.0),
        Err(KineticsError::InvalidConcentration {
            name: "A".to_string()
        })
    );
    let nand = [gate(DnaGateType::Nand, &["A", "B"], "Y")];
    assert_eq!(
        check_kinetics(&nand, &inputs, 10.0, 0.0),
        Err(KineticsError::InvalidStep)
    );
    assert_eq!(
        check_kinetics(&nand, &inputs, 10.0, 1.0),
        Err(KineticsError::InvalidConcentration {
            name: "A".to_string()
        })
    );
}
