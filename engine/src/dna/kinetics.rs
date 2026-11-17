// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust DNA circuit kinetic simulation

//! Mass-action kinetics of DNA gate circuits: RK4 or forward Euler with Arrhenius scaling.
//!
//! [`check_kinetics`] refuses a run it cannot simulate before anything is computed, and
//! [`simulate_kinetics`] runs the rest. The Python simulator in
//! `sc_neurocore.bridges.dna_simulation` makes the same refusals with the same words.

use std::collections::{HashMap, HashSet};

const R_GAS: f64 = 1.987e-3; // kcal/(mol·K)

/// Gate types for kinetic simulation.
///
/// Each kind sets the gate's effective rate `k_eff` from the input concentrations, which
/// are looked up by name (a missing name reads as 0) and scaled by `1e-9` before `k_hyb`;
/// the gate's `leak_rate` is then added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DnaGateType {
    /// `k_hyb * min(inputs) * 1e-9` when every input is above 0, else 0.
    And,
    /// `k_hyb * max(inputs) * 1e-9`.
    Or,
    /// `k_disp * (1 - min(input / max_conc, 1))`, from the first input.
    Not,
    /// `k_hyb * max(input - threshold * max_conc, 0) * 1e-9`, from the first input.
    Threshold,
    /// Inputs (select, a, b): `k_hyb * (s * a + (1 - s) * b) * 1e-9`, with
    /// `s = min(select / max_conc, 1)`.
    Mux,
    /// `k_hyb * input * 1e-9 * 5`, from the first input.
    Amplifier,
    /// `k_disp * min(input / max_conc, 1)`, from the first input.
    Buffer,
    /// No rate law here, so [`simulate_kinetics`] refuses it.
    Nand,
    /// No rate law here, so [`simulate_kinetics`] refuses it.
    Xor,
}

/// Every kind, in declaration order.
const GATE_KINDS: [DnaGateType; 9] = [
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

impl DnaGateType {
    /// The kind's name in upper case, as refusals and the Python binding write it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::And => "AND",
            Self::Or => "OR",
            Self::Not => "NOT",
            Self::Threshold => "THRESHOLD",
            Self::Mux => "MUX",
            Self::Amplifier => "AMPLIFIER",
            Self::Buffer => "BUFFER",
            Self::Nand => "NAND",
            Self::Xor => "XOR",
        }
    }

    /// The kind whose name is `name`, ignoring ASCII case, or `None` for any other name.
    pub fn parse(name: &str) -> Option<Self> {
        GATE_KINDS
            .into_iter()
            .find(|kind| kind.name().eq_ignore_ascii_case(name))
    }

    /// The fewest and most input names the kind reads, with the words a refusal uses, or
    /// `None` for a kind without a rate law.
    const fn input_rule(self) -> Option<(usize, usize, &'static str)> {
        match self {
            Self::And | Self::Or => Some((1, usize::MAX, "at least 1")),
            Self::Not | Self::Threshold | Self::Amplifier | Self::Buffer => {
                Some((1, 1, "exactly 1"))
            }
            Self::Mux => Some((3, 3, "exactly 3")),
            Self::Nand | Self::Xor => None,
        }
    }
}

/// A gate in the DNA circuit.
#[derive(Clone, Debug)]
pub struct DnaGateSpec {
    /// The gate's rate law.
    pub gate_type: DnaGateType,
    /// Names of the input species. `And` and `Or` read one or more, `Mux` exactly three as
    /// (select, a, b), and the single-input kinds exactly one; [`check_kinetics`] refuses
    /// another count.
    pub input_names: Vec<String>,
    /// Key of the gate's output trace in the simulation result.
    pub output_name: String,
    /// Fraction of `max_conc` the input must exceed; read by `Threshold` only.
    pub threshold: f64,
    /// Rate added to the gate's effective rate, for every kind (1/s).
    pub leak_rate: f64,
}

/// Kinetic simulation configuration.
pub struct KineticConfig {
    /// Hybridisation rate at 37 °C, per molar per second (concentrations are scaled by
    /// `1e-9`); scaled to `temperature_c` with an activation energy of 15 kcal/mol.
    pub k_hyb: f64,
    /// Displacement rate at 37 °C (1/s), scaled to `temperature_c` the same way.
    pub k_disp: f64,
    /// Operating temperature of the Arrhenius scaling (°C).
    pub temperature_c: f64,
    /// Ceiling every output approaches and is clamped to; `Not`, `Buffer` and `Mux` also
    /// divide an input by it.
    pub max_conc: f64,
    /// RK4 when true, forward Euler when false, for `dc/dt = k_eff (max_conc - c)`.
    pub use_rk4: bool,
}

impl Default for KineticConfig {
    fn default() -> Self {
        Self {
            k_hyb: 3e5,
            k_disp: 1.0,
            temperature_c: 37.0,
            max_conc: 200.0,
            use_rk4: true,
        }
    }
}

/// Arrhenius temperature scaling.
#[inline]
fn arrhenius_scale(k_ref: f64, temperature_c: f64, ea_kcal: f64) -> f64 {
    let t_ref = 310.15; // 37°C
    let t_op = temperature_c + 273.15;
    k_ref * (-(ea_kcal / R_GAS) * (1.0 / t_op - 1.0 / t_ref)).exp()
}

/// Compute effective rate constant for a gate.
fn compute_k_eff(gate: &DnaGateSpec, inputs: &HashMap<String, f64>, config: &KineticConfig) -> f64 {
    let k_hyb = arrhenius_scale(config.k_hyb, config.temperature_c, 15.0);
    let k_disp = arrhenius_scale(config.k_disp, config.temperature_c, 15.0);

    let k_eff = match gate.gate_type {
        DnaGateType::And => {
            let concs: Vec<f64> = gate
                .input_names
                .iter()
                .map(|n| *inputs.get(n).unwrap_or(&0.0))
                .collect();
            let all_present = concs.iter().all(|&c| c > 0.0);
            let min_c = concs.iter().cloned().fold(f64::MAX, f64::min);
            k_hyb * min_c * 1e-9 * if all_present { 1.0 } else { 0.0 }
        }
        DnaGateType::Or => {
            let concs: Vec<f64> = gate
                .input_names
                .iter()
                .map(|n| *inputs.get(n).unwrap_or(&0.0))
                .collect();
            let max_c = concs.iter().cloned().fold(0.0f64, f64::max);
            k_hyb * max_c * 1e-9
        }
        DnaGateType::Not => {
            let inp = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            k_disp * (1.0 - (inp / config.max_conc).min(1.0))
        }
        DnaGateType::Threshold => {
            let inp = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            let excess = (inp - gate.threshold * config.max_conc).max(0.0);
            k_hyb * excess * 1e-9
        }
        DnaGateType::Mux => {
            let sel = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            let a = *inputs.get(&gate.input_names[1]).unwrap_or(&0.0);
            let b = *inputs.get(&gate.input_names[2]).unwrap_or(&0.0);
            let sel_frac = (sel / config.max_conc).min(1.0);
            k_hyb * (sel_frac * a + (1.0 - sel_frac) * b) * 1e-9
        }
        DnaGateType::Amplifier => {
            let inp = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            k_hyb * inp * 1e-9 * 5.0
        }
        DnaGateType::Buffer => {
            let inp = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            k_disp * (inp / config.max_conc).min(1.0)
        }
        // Never reached: `check_kinetics` refuses the kinds without a rate law.
        DnaGateType::Nand | DnaGateType::Xor => 0.0,
    };

    k_eff + gate.leak_rate
}

/// The most samples a kinetic run may hold over all its traces, the time axis included.
///
/// A run of `n` steps and `g` gates holds `n * (g + 1)` values of 8 bytes; this bounds it to
/// 256 MiB, so a tiny `dt` is refused instead of asking for more memory than there is. The
/// Python simulator's `MAX_KINETIC_SAMPLES` is the same.
pub const MAX_KINETIC_SAMPLES: usize = 1 << 25;

/// Why [`check_kinetics`] refuses a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KineticsError {
    /// `dt` is not finite or not above 0.
    InvalidStep,
    /// `duration_s` is not finite or is shorter than one step, so the run would hold no sample.
    InvalidDuration,
    /// An input concentration is not finite or is below 0.
    InvalidConcentration {
        /// The input's name.
        name: String,
    },
    /// The gate's kind has no rate law here.
    NoRateLaw {
        /// The gate's position in the circuit.
        gate: usize,
        /// The gate's kind.
        kind: DnaGateType,
    },
    /// The gate's input name count does not fit its kind.
    InputCount {
        /// The gate's position in the circuit.
        gate: usize,
        /// The gate's kind.
        kind: DnaGateType,
        /// How many input names the gate has.
        count: usize,
        /// How many its kind needs: `"exactly 1"`, `"exactly 3"` or `"at least 1"`.
        needed: &'static str,
    },
    /// The gate's output is `"time"` or an earlier gate's output, whose trace it would replace.
    DuplicateOutput {
        /// The gate's position in the circuit.
        gate: usize,
        /// The output name.
        name: String,
    },
    /// The run would hold more than [`MAX_KINETIC_SAMPLES`] samples.
    TooManySamples,
}

impl std::fmt::Display for KineticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidStep => f.write_str("dt must be finite and above 0"),
            Self::InvalidDuration => f.write_str("duration_s must be finite and at least dt"),
            Self::InvalidConcentration { name } => write!(
                f,
                "input '{name}' must have a finite concentration of at least 0"
            ),
            Self::NoRateLaw { gate, kind } => write!(
                f,
                "gate {gate} ({}) has no rate law, so it cannot be simulated",
                kind.name()
            ),
            Self::InputCount {
                gate,
                kind,
                count,
                needed,
            } => write!(
                f,
                "gate {gate} ({}) has an input name count of {count}; its kind needs {needed}",
                kind.name()
            ),
            Self::DuplicateOutput { gate, name } => write!(
                f,
                "gate {gate}'s output '{name}' is already a trace of the result"
            ),
            Self::TooManySamples => write!(
                f,
                "the run would hold more than {MAX_KINETIC_SAMPLES} samples over its traces; \
                 use a larger dt or a shorter duration_s"
            ),
        }
    }
}

impl std::error::Error for KineticsError {}

/// Return a run's step count, `floor(duration_s / dt)`, or the reason it is refused.
///
/// # Errors
///
/// Checked in this order, so the first refused input or gate is the one named:
/// - [`KineticsError::InvalidStep`] if `dt` is not finite or not above 0;
/// - [`KineticsError::InvalidDuration`] if `duration_s` is not finite or is below `dt`;
/// - [`KineticsError::InvalidConcentration`] for the first input, by name in sorted order,
///   whose concentration is not finite or is below 0;
/// - for each gate in order, [`KineticsError::NoRateLaw`] for `Nand` and `Xor`,
///   [`KineticsError::InputCount`] for an input name count its kind does not read, and
///   [`KineticsError::DuplicateOutput`] for an output named `"time"` or an earlier gate's;
/// - [`KineticsError::TooManySamples`] if the steps times the traces exceed
///   [`MAX_KINETIC_SAMPLES`].
///
/// The Python simulator's `check_kinetics` makes the same checks with the same words.
pub fn check_kinetics(
    gates: &[DnaGateSpec],
    input_concentrations: &HashMap<String, f64>,
    duration_s: f64,
    dt: f64,
) -> Result<usize, KineticsError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(KineticsError::InvalidStep);
    }
    if !duration_s.is_finite() || duration_s < dt {
        return Err(KineticsError::InvalidDuration);
    }
    let mut names: Vec<&String> = input_concentrations.keys().collect();
    names.sort();
    for name in names {
        let concentration = input_concentrations[name];
        if !concentration.is_finite() || concentration < 0.0 {
            return Err(KineticsError::InvalidConcentration { name: name.clone() });
        }
    }
    let mut outputs: HashSet<&str> = HashSet::from(["time"]);
    for (index, gate) in gates.iter().enumerate() {
        let kind = gate.gate_type;
        let Some((fewest, most, needed)) = kind.input_rule() else {
            return Err(KineticsError::NoRateLaw { gate: index, kind });
        };
        let count = gate.input_names.len();
        if count < fewest || count > most {
            return Err(KineticsError::InputCount {
                gate: index,
                kind,
                count,
                needed,
            });
        }
        if !outputs.insert(gate.output_name.as_str()) {
            return Err(KineticsError::DuplicateOutput {
                gate: index,
                name: gate.output_name.clone(),
            });
        }
    }
    // The ratio is at least 1, but it may overflow to infinity.
    let steps = (duration_s / dt).floor();
    let samples = steps * (gates.len() + 1) as f64;
    if !samples.is_finite() || samples > MAX_KINETIC_SAMPLES as f64 {
        return Err(KineticsError::TooManySamples);
    }
    Ok(steps as usize)
}

/// Run kinetic simulation, returning time traces per output.
///
/// Each gate's `k_eff` is computed once from `input_concentrations`, and its output follows
/// `dc/dt = k_eff (max_conc - c)` from 0 in `floor(duration_s / dt)` samples, the first at 0,
/// each clamped to `[0, max_conc]`. The result holds `"time"`, sample `t` at `t * dt`, and one
/// trace per `output_name`.
///
/// # Errors
///
/// The run is checked by [`check_kinetics`] before anything is computed, and refused with its
/// error.
pub fn simulate_kinetics(
    gates: &[DnaGateSpec],
    input_concentrations: &HashMap<String, f64>,
    duration_s: f64,
    dt: f64,
    config: &KineticConfig,
) -> Result<HashMap<String, Vec<f64>>, KineticsError> {
    let n_steps = check_kinetics(gates, input_concentrations, duration_s, dt)?;
    let max_conc = config.max_conc;

    let mut result: HashMap<String, Vec<f64>> = HashMap::new();

    // Time axis
    let time: Vec<f64> = (0..n_steps).map(|t| t as f64 * dt).collect();
    result.insert("time".to_string(), time);

    for gate in gates {
        let k_eff = compute_k_eff(gate, input_concentrations, config);
        let mut conc = vec![0.0f64; n_steps];

        if config.use_rk4 {
            for t in 1..n_steps {
                let c = conc[t - 1];
                let k1 = k_eff * (max_conc - c) * dt;
                let k2 = k_eff * (max_conc - (c + k1 / 2.0)) * dt;
                let k3 = k_eff * (max_conc - (c + k2 / 2.0)) * dt;
                let k4 = k_eff * (max_conc - (c + k3)) * dt;
                conc[t] = (c + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
                    .max(0.0)
                    .min(max_conc);
            }
        } else {
            for t in 1..n_steps {
                let d = k_eff * (max_conc - conc[t - 1]) * dt;
                conc[t] = (conc[t - 1] + d).max(0.0).min(max_conc);
            }
        }

        result.insert(gate.output_name.clone(), conc);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_kinetics_and_gate() {
        let gates = vec![DnaGateSpec {
            gate_type: DnaGateType::And,
            input_names: vec!["A".to_string(), "B".to_string()],
            output_name: "C".to_string(),
            threshold: 0.5,
            leak_rate: 1e-7,
        }];
        let mut inputs = HashMap::new();
        inputs.insert("A".to_string(), 200.0);
        inputs.insert("B".to_string(), 200.0);

        let result = simulate_kinetics(&gates, &inputs, 1800.0, 1.0, &KineticConfig::default())
            .expect("an AND gate with both inputs is simulated");
        let c = result.get("C").unwrap();
        assert!(c.last().unwrap() > &50.0);
    }

    #[test]
    fn test_arrhenius_higher_temp_faster() {
        let k37 = arrhenius_scale(3e5, 37.0, 15.0);
        let k25 = arrhenius_scale(3e5, 25.0, 15.0);
        assert!(k37 > k25);
    }
}
