// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — modular Rust execution for SC-COMPTE-WM-NETWORK

//! Full 2,560-cell Rust runtime for the SC Compte working-memory network.
//!
//! This is the native implementation of the separately named SC project
//! network. It does not replace the preserved scalar Compte cell. The module
//! mirrors the Python v1 contract: counter-addressed aggregate Poisson input,
//! source-unit midpoint RK2, no recurrent autapses, circular E→E connectivity,
//! optional structured E→I connectivity, and explicit-event co-simulation.

use rustfft::{num_complex::Complex64, Fft, FftPlanner};
use std::sync::Arc;

/// Fixed excitatory population size.
pub const N_EXCITATORY: usize = 2048;
/// Fixed inhibitory population size.
pub const N_INHIBITORY: usize = 512;
/// Fixed source timestep in milliseconds.
pub const DT_MS: f64 = 0.02;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const STEP_MIX: u64 = 0xD1B5_4A32_D192_ED03;
const STREAM_MIX: u64 = 0x94D0_49BB_1331_11EB;
const V_MIN: f64 = -200.0;
const V_MAX: f64 = 100.0;
const GATE_MAX: f64 = 1.0e6;

/// Frozen native configuration corresponding to the Python v1 specification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SCCompteWMNetworkSpec {
    /// Counter-stream seed.
    pub seed: u64,
    /// Whether E→I uses the tuned circular footprint.
    pub structured_ei: bool,
    /// Whether the modulated 1.2× NMDA and 1.4× GABAA set is selected.
    pub modulated: bool,
    /// Whether recurrent E→E and I→I self-connections are retained.
    pub allow_recurrent_autapses: bool,
}

impl Default for SCCompteWMNetworkSpec {
    fn default() -> Self {
        Self {
            seed: 42,
            structured_ei: false,
            modulated: false,
            allow_recurrent_autapses: false,
        }
    }
}

/// Complete native dynamic state in the same array order as Python.
#[derive(Clone, Debug, PartialEq)]
pub struct SCCompteWMNetworkState {
    /// Absolute counter-stream step.
    pub step_index: u64,
    /// Pyramidal membrane voltages in mV.
    pub v_exc_mv: Vec<f64>,
    /// Interneuron membrane voltages in mV.
    pub v_inh_mv: Vec<f64>,
    /// Pyramidal refractory time remaining in ms.
    pub refractory_exc_ms: Vec<f64>,
    /// Interneuron refractory time remaining in ms.
    pub refractory_inh_ms: Vec<f64>,
    /// Per-pyramidal-cell external AMPA gates.
    pub external_ampa_exc: Vec<f64>,
    /// Per-interneuron external AMPA gates.
    pub external_ampa_inh: Vec<f64>,
    /// Presynaptic pyramidal NMDA open fractions.
    pub recurrent_nmda: Vec<f64>,
    /// Presynaptic pyramidal NMDA rise precursors.
    pub recurrent_nmda_rise: Vec<f64>,
    /// Presynaptic interneuron GABAA gates.
    pub recurrent_gabaa: Vec<f64>,
}

impl Default for SCCompteWMNetworkState {
    fn default() -> Self {
        Self {
            step_index: 0,
            v_exc_mv: vec![-70.0; N_EXCITATORY],
            v_inh_mv: vec![-70.0; N_INHIBITORY],
            refractory_exc_ms: vec![0.0; N_EXCITATORY],
            refractory_inh_ms: vec![0.0; N_INHIBITORY],
            external_ampa_exc: vec![0.0; N_EXCITATORY],
            external_ampa_inh: vec![0.0; N_INHIBITORY],
            recurrent_nmda: vec![0.0; N_EXCITATORY],
            recurrent_nmda_rise: vec![0.0; N_EXCITATORY],
            recurrent_gabaa: vec![0.0; N_INHIBITORY],
        }
    }
}

/// Event receipt emitted by one successful atomic native step.
#[derive(Clone, Debug, PartialEq)]
pub struct SCCompteWMStepReceipt {
    /// Step address consumed by the transition.
    pub step_index: u64,
    /// Pyramidal sampled output events.
    pub excitatory_spikes: Vec<bool>,
    /// Interneuron sampled output events.
    pub inhibitory_spikes: Vec<bool>,
    /// Aggregate external events delivered to pyramidal cells.
    pub excitatory_input_events: u64,
    /// Aggregate external events delivered to interneurons.
    pub inhibitory_input_events: u64,
}

/// Return one portable counter-addressed Poisson sample for every cell.
pub fn counter_poisson_counts(
    population_size: usize,
    rate_hz: f64,
    dt_ms: f64,
    seed: u64,
    stream: u64,
    step_index: u64,
) -> Result<Vec<u64>, &'static str> {
    if population_size == 0
        || !rate_hz.is_finite()
        || rate_hz < 0.0
        || !dt_ms.is_finite()
        || dt_ms <= 0.0
    {
        return Err("invalid counter-Poisson configuration");
    }
    let mean = rate_hz * dt_ms / 1000.0;
    if mean > 32.0 {
        return Err("counter-Poisson mean exceeds safety envelope");
    }
    let mut probability = (-mean).exp();
    let mut cumulative = probability;
    let mut cdf = vec![cumulative];
    let mut count = 0_u64;
    while cumulative < 1.0 - 1.0e-15 {
        count += 1;
        if count > 255 {
            return Err("counter-Poisson inverse CDF exceeded event range");
        }
        probability *= mean / count as f64;
        cumulative += probability;
        cdf.push(cumulative.min(1.0));
    }
    let last = cdf.last_mut().ok_or("counter-Poisson CDF is empty")?;
    *last = 1.0;
    Ok((0..population_size)
        .map(|cell| {
            let counter = seed
                .wrapping_add(step_index.wrapping_mul(STEP_MIX))
                .wrapping_add(stream.wrapping_mul(STREAM_MIX))
                .wrapping_add((cell as u64).wrapping_mul(GOLDEN));
            let bits = splitmix64(counter);
            let uniform = ((bits >> 11) as f64 + 0.5) * 2.0_f64.powi(-53);
            cdf.partition_point(|threshold| *threshold < uniform) as u64
        })
        .collect())
}

fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Full native executor with preplanned FFTs and fail-closed state transitions.
pub struct SCCompteWMNetwork {
    /// Frozen runtime configuration.
    pub spec: SCCompteWMNetworkSpec,
    /// Complete public native state.
    pub state: SCCompteWMNetworkState,
    ee_kernel: Vec<f64>,
    ee_spectrum: Vec<Complex64>,
    ei_spectrum: Option<Vec<Complex64>>,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl SCCompteWMNetwork {
    /// Construct the fixed network and validate an optional checkpoint state.
    pub fn new(
        spec: SCCompteWMNetworkSpec,
        state: Option<SCCompteWMNetworkState>,
    ) -> Result<Self, &'static str> {
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(N_EXCITATORY);
        let inverse = planner.plan_fft_inverse(N_EXCITATORY);
        let ee_kernel = footprint(1.62, 18.0);
        let ee_spectrum = spectrum(&ee_kernel, &forward);
        let ei_spectrum = spec
            .structured_ei
            .then(|| spectrum(&footprint(1.25, 18.0), &forward));
        let network = Self {
            spec,
            state: state.unwrap_or_default(),
            ee_kernel,
            ee_spectrum,
            ei_spectrum,
            forward,
            inverse,
        };
        network.validate_state(&network.state)?;
        Ok(network)
    }

    /// Restore the zero-gate leak-equilibrium state and counter address zero.
    pub fn reset(&mut self) {
        self.state = SCCompteWMNetworkState::default();
    }

    /// Advance using the canonical counter-addressed external input streams.
    pub fn step(
        &mut self,
        direct_exc_current_pa: &[f64],
    ) -> Result<SCCompteWMStepReceipt, &'static str> {
        let exc = counter_poisson_counts(
            N_EXCITATORY,
            1800.0,
            DT_MS,
            self.spec.seed,
            0,
            self.state.step_index,
        )?;
        let inh = counter_poisson_counts(
            N_INHIBITORY,
            1800.0,
            DT_MS,
            self.spec.seed,
            1,
            self.state.step_index,
        )?;
        self.step_with_events(direct_exc_current_pa, &exc, &inh)
    }

    /// Advance with explicit per-cell external counts for co-simulation.
    pub fn step_with_events(
        &mut self,
        direct_exc_current_pa: &[f64],
        external_exc_events: &[u64],
        external_inh_events: &[u64],
    ) -> Result<SCCompteWMStepReceipt, &'static str> {
        self.validate_state(&self.state)?;
        if direct_exc_current_pa.len() != N_EXCITATORY
            || external_exc_events.len() != N_EXCITATORY
            || external_inh_events.len() != N_INHIBITORY
            || !direct_exc_current_pa.iter().all(|value| value.is_finite())
        {
            return Err("invalid SC Compte network step input");
        }
        let mut start = Stage {
            v_exc: self.state.v_exc_mv.clone(),
            v_inh: self.state.v_inh_mv.clone(),
            ext_exc: add_events(&self.state.external_ampa_exc, external_exc_events)?,
            ext_inh: add_events(&self.state.external_ampa_inh, external_inh_events)?,
            nmda: self.state.recurrent_nmda.clone(),
            nmda_rise: self.state.recurrent_nmda_rise.clone(),
            gabaa: self.state.recurrent_gabaa.clone(),
        };
        let active_exc: Vec<bool> = self
            .state
            .refractory_exc_ms
            .iter()
            .map(|value| *value <= 0.0)
            .collect();
        let active_inh: Vec<bool> = self
            .state
            .refractory_inh_ms
            .iter()
            .map(|value| *value <= 0.0)
            .collect();
        let currents_na: Vec<f64> = direct_exc_current_pa.iter().map(|v| v / 1000.0).collect();
        let k1 = self.derivatives(&start, &currents_na, &active_exc, &active_inh);
        let midpoint = start.add_scaled(&k1, 0.5 * DT_MS);
        let k2 = self.derivatives(&midpoint, &currents_na, &active_exc, &active_inh);
        start.add_scaled_in_place(&k2, DT_MS);

        let mut ref_exc: Vec<f64> = self
            .state
            .refractory_exc_ms
            .iter()
            .map(|value| (value - DT_MS).max(0.0))
            .collect();
        let mut ref_inh: Vec<f64> = self
            .state
            .refractory_inh_ms
            .iter()
            .map(|value| (value - DT_MS).max(0.0))
            .collect();
        let mut exc_spikes = vec![false; N_EXCITATORY];
        let mut inh_spikes = vec![false; N_INHIBITORY];
        for index in 0..N_EXCITATORY {
            if !active_exc[index] {
                start.v_exc[index] = -60.0;
            } else if start.v_exc[index] >= -50.0 {
                start.v_exc[index] = -60.0;
                ref_exc[index] = 2.0;
                exc_spikes[index] = true;
                start.nmda_rise[index] += 1.0;
            }
        }
        for index in 0..N_INHIBITORY {
            if !active_inh[index] {
                start.v_inh[index] = -60.0;
            } else if start.v_inh[index] >= -50.0 {
                start.v_inh[index] = -60.0;
                ref_inh[index] = 1.0;
                inh_spikes[index] = true;
                start.gabaa[index] += 1.0;
            }
        }
        let next = SCCompteWMNetworkState {
            step_index: self
                .state
                .step_index
                .checked_add(1)
                .ok_or("step counter overflow")?,
            v_exc_mv: start.v_exc,
            v_inh_mv: start.v_inh,
            refractory_exc_ms: ref_exc,
            refractory_inh_ms: ref_inh,
            external_ampa_exc: start.ext_exc,
            external_ampa_inh: start.ext_inh,
            recurrent_nmda: start.nmda,
            recurrent_nmda_rise: start.nmda_rise,
            recurrent_gabaa: start.gabaa,
        };
        self.validate_state(&next)?;
        let receipt = SCCompteWMStepReceipt {
            step_index: self.state.step_index,
            excitatory_spikes: exc_spikes,
            inhibitory_spikes: inh_spikes,
            excitatory_input_events: external_exc_events.iter().sum(),
            inhibitory_input_events: external_inh_events.iter().sum(),
        };
        self.state = next;
        Ok(receipt)
    }

    fn derivatives(
        &self,
        stage: &Stage,
        currents_na: &[f64],
        active_exc: &[bool],
        active_inh: &[bool],
    ) -> Stage {
        let (ee, ei, ie, ii) = self.aggregates(&stage.nmda, &stage.gabaa);
        let nmda_scale = if self.spec.modulated { 1.2 } else { 1.0 };
        let gaba_scale = if self.spec.modulated { 1.4 } else { 1.0 };
        let mut result = Stage::zeros();
        for index in 0..N_EXCITATORY {
            let v = stage.v_exc[index];
            result.v_exc[index] = if active_exc[index] {
                (-0.025 * (v + 70.0)
                    - 0.0031 * stage.ext_exc[index] * v
                    - 0.000_381 * nmda_scale * ee[index] * mg_block(v) * v
                    - 0.001_336 * gaba_scale * ie[index] * (v + 70.0)
                    + currents_na[index])
                    / 0.5
            } else {
                0.0
            };
            result.ext_exc[index] = -stage.ext_exc[index] / 2.0;
            result.nmda[index] = -stage.nmda[index] / 100.0
                + 0.5 * stage.nmda_rise[index] * (1.0 - stage.nmda[index]);
            result.nmda_rise[index] = -stage.nmda_rise[index] / 2.0;
        }
        for index in 0..N_INHIBITORY {
            let v = stage.v_inh[index];
            result.v_inh[index] = if active_inh[index] {
                (-0.020 * (v + 70.0)
                    - 0.00238 * stage.ext_inh[index] * v
                    - 0.000_292 * nmda_scale * ei[index] * mg_block(v) * v
                    - 0.001_024 * gaba_scale * ii[index] * (v + 70.0))
                    / 0.2
            } else {
                0.0
            };
            result.ext_inh[index] = -stage.ext_inh[index] / 2.0;
            result.gabaa[index] = -stage.gabaa[index] / 10.0;
        }
        result
    }

    fn aggregates(&self, nmda: &[f64], gabaa: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut ee = self.circular_sum(nmda, &self.ee_spectrum);
        if !self.spec.allow_recurrent_autapses {
            for index in 0..N_EXCITATORY {
                ee[index] -= self.ee_kernel[0] * nmda[index];
            }
        }
        let ei = if let Some(spectrum) = &self.ei_spectrum {
            self.circular_sum(nmda, spectrum)
                .into_iter()
                .step_by(4)
                .collect()
        } else {
            vec![nmda.iter().sum(); N_INHIBITORY]
        };
        let total: f64 = gabaa.iter().sum();
        let ie = vec![total; N_EXCITATORY];
        let mut ii = vec![total; N_INHIBITORY];
        if !self.spec.allow_recurrent_autapses {
            for index in 0..N_INHIBITORY {
                ii[index] -= gabaa[index];
            }
        }
        (ee, ei, ie, ii)
    }

    fn circular_sum(&self, source: &[f64], kernel_spectrum: &[Complex64]) -> Vec<f64> {
        let mut buffer: Vec<Complex64> = source
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect();
        self.forward.process(&mut buffer);
        for (value, kernel) in buffer.iter_mut().zip(kernel_spectrum) {
            *value *= kernel;
        }
        self.inverse.process(&mut buffer);
        buffer
            .into_iter()
            .map(|value| value.re / N_EXCITATORY as f64)
            .collect()
    }

    fn validate_state(&self, state: &SCCompteWMNetworkState) -> Result<(), &'static str> {
        let valid = state.v_exc_mv.len() == N_EXCITATORY
            && state.v_inh_mv.len() == N_INHIBITORY
            && state.refractory_exc_ms.len() == N_EXCITATORY
            && state.refractory_inh_ms.len() == N_INHIBITORY
            && state.external_ampa_exc.len() == N_EXCITATORY
            && state.external_ampa_inh.len() == N_INHIBITORY
            && state.recurrent_nmda.len() == N_EXCITATORY
            && state.recurrent_nmda_rise.len() == N_EXCITATORY
            && state.recurrent_gabaa.len() == N_INHIBITORY
            && state
                .v_exc_mv
                .iter()
                .chain(&state.v_inh_mv)
                .all(|v| v.is_finite() && (V_MIN..=V_MAX).contains(v))
            && state
                .refractory_exc_ms
                .iter()
                .chain(&state.refractory_inh_ms)
                .chain(&state.external_ampa_exc)
                .chain(&state.external_ampa_inh)
                .chain(&state.recurrent_nmda_rise)
                .chain(&state.recurrent_gabaa)
                .all(|v| v.is_finite() && (0.0..=GATE_MAX).contains(v))
            && state
                .recurrent_nmda
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v));
        valid.then_some(()).ok_or("invalid SC Compte network state")
    }
}

struct Stage {
    v_exc: Vec<f64>,
    v_inh: Vec<f64>,
    ext_exc: Vec<f64>,
    ext_inh: Vec<f64>,
    nmda: Vec<f64>,
    nmda_rise: Vec<f64>,
    gabaa: Vec<f64>,
}

impl Stage {
    fn zeros() -> Self {
        Self {
            v_exc: vec![0.0; N_EXCITATORY],
            v_inh: vec![0.0; N_INHIBITORY],
            ext_exc: vec![0.0; N_EXCITATORY],
            ext_inh: vec![0.0; N_INHIBITORY],
            nmda: vec![0.0; N_EXCITATORY],
            nmda_rise: vec![0.0; N_EXCITATORY],
            gabaa: vec![0.0; N_INHIBITORY],
        }
    }

    fn add_scaled(&self, derivative: &Self, scale: f64) -> Self {
        let mut result = Self::zeros();
        result.assign_scaled(self, derivative, scale);
        result
    }

    fn add_scaled_in_place(&mut self, derivative: &Self, scale: f64) {
        let base = Self {
            v_exc: self.v_exc.clone(),
            v_inh: self.v_inh.clone(),
            ext_exc: self.ext_exc.clone(),
            ext_inh: self.ext_inh.clone(),
            nmda: self.nmda.clone(),
            nmda_rise: self.nmda_rise.clone(),
            gabaa: self.gabaa.clone(),
        };
        self.assign_scaled(&base, derivative, scale);
    }

    fn assign_scaled(&mut self, base: &Self, derivative: &Self, scale: f64) {
        fn assign(out: &mut [f64], base: &[f64], derivative: &[f64], scale: f64) {
            for index in 0..out.len() {
                out[index] = base[index] + scale * derivative[index];
            }
        }
        assign(&mut self.v_exc, &base.v_exc, &derivative.v_exc, scale);
        assign(&mut self.v_inh, &base.v_inh, &derivative.v_inh, scale);
        assign(&mut self.ext_exc, &base.ext_exc, &derivative.ext_exc, scale);
        assign(&mut self.ext_inh, &base.ext_inh, &derivative.ext_inh, scale);
        assign(&mut self.nmda, &base.nmda, &derivative.nmda, scale);
        assign(
            &mut self.nmda_rise,
            &base.nmda_rise,
            &derivative.nmda_rise,
            scale,
        );
        assign(&mut self.gabaa, &base.gabaa, &derivative.gabaa, scale);
    }
}

fn add_events(gates: &[f64], events: &[u64]) -> Result<Vec<f64>, &'static str> {
    gates
        .iter()
        .zip(events)
        .map(|(gate, event)| {
            let value = gate + *event as f64;
            (value.is_finite() && value <= GATE_MAX)
                .then_some(value)
                .ok_or("external event gate exceeds safety envelope")
        })
        .collect()
}

fn mg_block(voltage: f64) -> f64 {
    1.0 / (1.0 + (-0.062 * voltage).exp() / 3.57)
}

fn footprint(j_plus: f64, sigma_deg: f64) -> Vec<f64> {
    let gaussian: Vec<f64> = (0..N_EXCITATORY)
        .map(|index| {
            let angle = index as f64 * 360.0 / N_EXCITATORY as f64;
            let distance = (angle + 180.0).rem_euclid(360.0) - 180.0;
            (-0.5 * (distance / sigma_deg).powi(2)).exp()
        })
        .collect();
    let mean = gaussian.iter().sum::<f64>() / N_EXCITATORY as f64;
    let j_minus = (1.0 - j_plus * mean) / (1.0 - mean);
    let mut weights: Vec<f64> = gaussian
        .iter()
        .map(|g| j_minus + (j_plus - j_minus) * g)
        .collect();
    let weight_mean = weights.iter().sum::<f64>() / N_EXCITATORY as f64;
    for weight in &mut weights {
        *weight /= weight_mean;
    }
    weights
}

fn spectrum(values: &[f64], fft: &Arc<dyn Fft<f64>>) -> Vec<Complex64> {
    let mut result: Vec<Complex64> = values
        .iter()
        .map(|value| Complex64::new(*value, 0.0))
        .collect();
    fft.process(&mut result);
    result
}
