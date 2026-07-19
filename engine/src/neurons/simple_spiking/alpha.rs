// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dual alpha-synapse leaky integrate-and-fire engine

//! Exact piecewise-constant-input flow for the dual alpha-synapse LIF: a leaky
//! membrane relaxation driven by two five-state alpha-filter cascades, with
//! the exact alpha-current convolution including the equal-time-constant limit.

use std::error::Error;
use std::fmt;

/// Typed caller-contract and numerical failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlphaError {
    /// A state or parameter is not finite.
    NonFiniteConfiguration,
    /// A magnitude or ordering constraint is violated.
    InvalidScale,
    /// One external-current value is not finite.
    NonFiniteInput,
    /// The exact-flow candidate is not finite.
    NonFiniteCandidate,
}

impl fmt::Display for AlphaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFiniteConfiguration => "alpha state and parameters must be finite",
            Self::InvalidScale => {
                "alpha tau_v/tau_exc/tau_inh/dt must be positive and v_threshold must exceed v_rest"
            }
            Self::NonFiniteInput => "alpha input must contain only finite values",
            Self::NonFiniteCandidate => "alpha exact-flow candidate must remain finite",
        };
        formatter.write_str(message)
    }
}

impl Error for AlphaError {}

/// Dual excitatory/inhibitory alpha-synapse leaky integrate-and-fire neuron.
#[derive(Clone, Debug)]
pub struct AlphaNeuron {
    /// Membrane potential.
    pub v: f64,
    /// Excitatory alpha-rise state.
    pub a_exc: f64,
    /// Excitatory synaptic current.
    pub i_exc: f64,
    /// Inhibitory alpha-rise state.
    pub a_inh: f64,
    /// Inhibitory synaptic current.
    pub i_inh: f64,
    /// Leak reversal potential, also the somatic spike reset.
    pub v_rest: f64,
    /// Spike threshold; must exceed `v_rest`.
    pub v_threshold: f64,
    /// Positive membrane time constant.
    pub tau_v: f64,
    /// Positive excitatory alpha time constant.
    pub tau_exc: f64,
    /// Positive inhibitory alpha time constant.
    pub tau_inh: f64,
    /// Positive piecewise-constant-input sampling interval.
    pub dt: f64,
}

impl AlphaNeuron {
    /// Construct the catalogue model-family defaults.
    pub fn new() -> Self {
        Self {
            v: 0.0,
            a_exc: 0.0,
            i_exc: 0.0,
            a_inh: 0.0,
            i_inh: 0.0,
            v_rest: 0.0,
            v_threshold: 1.0,
            tau_v: 20.0,
            tau_exc: 5.0,
            tau_inh: 10.0,
            dt: 1.0,
        }
    }

    /// Construct and validate a complete numerical configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_parameters(
        v: f64,
        a_exc: f64,
        i_exc: f64,
        a_inh: f64,
        i_inh: f64,
        v_rest: f64,
        v_threshold: f64,
        tau_v: f64,
        tau_exc: f64,
        tau_inh: f64,
        dt: f64,
    ) -> Result<Self, AlphaError> {
        let neuron = Self {
            v,
            a_exc,
            i_exc,
            a_inh,
            i_inh,
            v_rest,
            v_threshold,
            tau_v,
            tau_exc,
            tau_inh,
            dt,
        };
        neuron.validate()?;
        Ok(neuron)
    }

    fn validate(&self) -> Result<(), AlphaError> {
        if ![
            self.v,
            self.a_exc,
            self.i_exc,
            self.a_inh,
            self.i_inh,
            self.v_rest,
            self.v_threshold,
            self.tau_v,
            self.tau_exc,
            self.tau_inh,
            self.dt,
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            return Err(AlphaError::NonFiniteConfiguration);
        }
        if self.tau_v <= 0.0
            || self.tau_exc <= 0.0
            || self.tau_inh <= 0.0
            || self.dt <= 0.0
            || self.v_threshold <= self.v_rest
        {
            return Err(AlphaError::InvalidScale);
        }
        Ok(())
    }

    fn filter_candidates(
        rise_state: f64,
        current_state: f64,
        drive: f64,
        tau: f64,
        dt: f64,
    ) -> Result<(f64, f64), AlphaError> {
        let steady_state = tau * drive;
        let rise_delta = rise_state - steady_state;
        let current_delta = current_state - steady_state;
        let decay = (-dt / tau).exp();
        let rise_next = steady_state + rise_delta * decay;
        let current_next = steady_state + decay * (current_delta + rise_delta * dt / tau);
        if !rise_next.is_finite() || !current_next.is_finite() {
            return Err(AlphaError::NonFiniteCandidate);
        }
        Ok((rise_next, current_next))
    }

    fn drive_contribution(
        current_delta: f64,
        rise_delta: f64,
        tau_drive: f64,
        tau_v: f64,
        dt: f64,
    ) -> Result<f64, AlphaError> {
        let rate_v = 1.0 / tau_v;
        let rate_drive = 1.0 / tau_drive;
        let decay_v = (-dt / tau_v).exp();
        let decay_drive = (-dt / tau_drive).exp();
        let contribution = if (rate_v - rate_drive).abs() <= 1.0e-14 {
            rate_v * decay_v * (current_delta * dt + rise_delta * dt * dt / (2.0 * tau_drive))
        } else {
            let rate_delta = rate_v - rate_drive;
            let first_order = current_delta * (decay_drive - decay_v) / rate_delta;
            let second_order = rise_delta / tau_drive
                * (decay_drive * (rate_delta * dt - 1.0) + decay_v)
                / (rate_delta * rate_delta);
            rate_v * (first_order + second_order)
        };
        if !contribution.is_finite() {
            return Err(AlphaError::NonFiniteCandidate);
        }
        Ok(contribution)
    }

    /// Advance one exact-flow interval with explicit error reporting.
    pub fn try_step(&mut self, exc_current: f64, inh_current: f64) -> Result<i32, AlphaError> {
        self.validate()?;
        if !exc_current.is_finite() || !inh_current.is_finite() {
            return Err(AlphaError::NonFiniteInput);
        }
        let (a_exc_next, i_exc_next) =
            Self::filter_candidates(self.a_exc, self.i_exc, exc_current, self.tau_exc, self.dt)?;
        let (a_inh_next, i_inh_next) =
            Self::filter_candidates(self.a_inh, self.i_inh, inh_current, self.tau_inh, self.dt)?;
        let exc_steady = self.tau_exc * exc_current;
        let inh_steady = self.tau_inh * inh_current;
        let v_steady = self.v_rest + exc_steady - inh_steady;
        let decay_v = (-self.dt / self.tau_v).exp();
        let v_next = v_steady
            + (self.v - v_steady) * decay_v
            + Self::drive_contribution(
                self.i_exc - exc_steady,
                self.a_exc - exc_steady,
                self.tau_exc,
                self.tau_v,
                self.dt,
            )?
            - Self::drive_contribution(
                self.i_inh - inh_steady,
                self.a_inh - inh_steady,
                self.tau_inh,
                self.tau_v,
                self.dt,
            )?;
        if !v_next.is_finite() {
            return Err(AlphaError::NonFiniteCandidate);
        }
        self.a_exc = a_exc_next;
        self.i_exc = i_exc_next;
        self.a_inh = a_inh_next;
        self.i_inh = i_inh_next;
        if v_next >= self.v_threshold {
            self.v = self.v_rest;
            return Ok(1);
        }
        self.v = v_next;
        Ok(0)
    }

    /// Preserve the legacy scalar API while failing closed on invalid input.
    pub fn step(&mut self, exc_current: f64, inh_current: f64) -> i32 {
        self.try_step(exc_current, inh_current).unwrap_or(0)
    }

    /// Restore the documented rest state while preserving configuration.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.a_exc = 0.0;
        self.i_exc = 0.0;
        self.a_inh = 0.0;
        self.i_inh = 0.0;
    }
}

impl Default for AlphaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete post-update trace and final-state receipt.
pub struct AlphaTrace {
    /// Membrane-potential trace.
    pub v: Vec<f64>,
    /// Excitatory rise-state trace.
    pub a_exc: Vec<f64>,
    /// Excitatory current trace.
    pub i_exc: Vec<f64>,
    /// Inhibitory rise-state trace.
    pub a_inh: Vec<f64>,
    /// Inhibitory current trace.
    pub i_inh: Vec<f64>,
    /// Binary spike-event trace represented as floating point for ABI parity.
    pub spikes: Vec<f64>,
    /// Final ``[v, a_exc, i_exc, a_inh, i_inh]`` state, including the initial
    /// state for an empty batch.
    pub final_state: [f64; 5],
    /// Number of candidate-crossing spikes.
    pub spike_count: usize,
}

/// Simulate a complete caller-owned piecewise-constant drive batch atomically.
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    v: f64,
    a_exc: f64,
    i_exc: f64,
    a_inh: f64,
    i_inh: f64,
    v_rest: f64,
    v_threshold: f64,
    tau_v: f64,
    tau_exc: f64,
    tau_inh: f64,
    dt: f64,
    exc_current: &[f64],
    inh_current: &[f64],
) -> Result<AlphaTrace, AlphaError> {
    if exc_current.len() != inh_current.len() {
        return Err(AlphaError::NonFiniteInput);
    }
    let mut neuron = AlphaNeuron::with_parameters(
        v,
        a_exc,
        i_exc,
        a_inh,
        i_inh,
        v_rest,
        v_threshold,
        tau_v,
        tau_exc,
        tau_inh,
        dt,
    )?;
    if !exc_current.iter().all(|value| value.is_finite())
        || !inh_current.iter().all(|value| value.is_finite())
    {
        return Err(AlphaError::NonFiniteInput);
    }
    let steps = exc_current.len();
    let mut v_trace = Vec::with_capacity(steps);
    let mut a_exc_trace = Vec::with_capacity(steps);
    let mut i_exc_trace = Vec::with_capacity(steps);
    let mut a_inh_trace = Vec::with_capacity(steps);
    let mut i_inh_trace = Vec::with_capacity(steps);
    let mut spikes = Vec::with_capacity(steps);
    let mut spike_count = 0usize;
    for index in 0..steps {
        let spike = neuron.try_step(exc_current[index], inh_current[index])?;
        spike_count += spike as usize;
        v_trace.push(neuron.v);
        a_exc_trace.push(neuron.a_exc);
        i_exc_trace.push(neuron.i_exc);
        a_inh_trace.push(neuron.a_inh);
        i_inh_trace.push(neuron.i_inh);
        spikes.push(f64::from(spike));
    }
    Ok(AlphaTrace {
        v: v_trace,
        a_exc: a_exc_trace,
        i_exc: i_exc_trace,
        a_inh: a_inh_trace,
        i_inh: i_inh_trace,
        spikes,
        final_state: [
            neuron.v,
            neuron.a_exc,
            neuron.i_exc,
            neuron.a_inh,
            neuron.i_inh,
        ],
        spike_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_catalogue_model_family() {
        let neuron = AlphaNeuron::new();
        assert_eq!(
            (
                neuron.v,
                neuron.a_exc,
                neuron.i_exc,
                neuron.a_inh,
                neuron.i_inh,
                neuron.v_rest,
                neuron.v_threshold,
                neuron.tau_v,
                neuron.tau_exc,
                neuron.tau_inh,
                neuron.dt,
            ),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 5.0, 10.0, 1.0)
        );
    }

    #[test]
    fn filter_matches_exact_alpha_cascade() {
        let (rise_next, current_next) =
            AlphaNeuron::filter_candidates(0.25, 0.1, 2.0, 5.0, 0.5).unwrap();
        let steady = 5.0 * 2.0;
        let decay = (-0.5_f64 / 5.0).exp();
        let expected_rise = steady + (0.25 - steady) * decay;
        let expected_current = steady + decay * ((0.1 - steady) + (0.25 - steady) * 0.5 / 5.0);
        assert!((rise_next - expected_rise).abs() < 1.0e-12);
        assert!((current_next - expected_current).abs() < 1.0e-12);
    }

    #[test]
    fn drive_contribution_handles_equal_time_constants() {
        let exact = AlphaNeuron::drive_contribution(0.3, 0.2, 20.0, 20.0, 0.5).unwrap();
        let rate = 1.0 / 20.0;
        let decay = (-0.5_f64 / 20.0).exp();
        let expected = rate * decay * (0.3 * 0.5 + 0.2 * 0.5 * 0.5 / (2.0 * 20.0));
        assert!((exact - expected).abs() < 1.0e-12);
    }

    #[test]
    fn spike_resets_only_the_membrane() {
        let mut neuron = AlphaNeuron {
            v: 0.9,
            a_exc: 0.4,
            i_exc: 0.6,
            a_inh: 0.2,
            i_inh: 0.1,
            v_threshold: 0.5,
            ..AlphaNeuron::new()
        };
        let (a_exc_before, i_exc_before, a_inh_before, i_inh_before) =
            (neuron.a_exc, neuron.i_exc, neuron.a_inh, neuron.i_inh);
        assert_eq!(neuron.try_step(0.0, 0.0), Ok(1));
        assert_eq!(neuron.v, 0.0);
        let decay_exc = (-1.0_f64 / 5.0).exp();
        let decay_inh = (-1.0_f64 / 10.0).exp();
        assert!((neuron.a_exc - a_exc_before * decay_exc).abs() < 1.0e-12);
        assert!(
            (neuron.i_exc - decay_exc * (i_exc_before + a_exc_before * 1.0 / 5.0)).abs() < 1.0e-12
        );
        assert!((neuron.a_inh - a_inh_before * decay_inh).abs() < 1.0e-12);
        assert!(
            (neuron.i_inh - decay_inh * (i_inh_before + a_inh_before * 1.0 / 10.0)).abs() < 1.0e-12
        );
    }

    #[test]
    fn invalid_step_is_atomic() {
        let mut neuron = AlphaNeuron::new();
        neuron.v = 0.25;
        neuron.a_exc = 0.5;
        let before = (
            neuron.v,
            neuron.a_exc,
            neuron.i_exc,
            neuron.a_inh,
            neuron.i_inh,
        );
        assert_eq!(
            neuron.try_step(f64::NAN, 0.0),
            Err(AlphaError::NonFiniteInput)
        );
        assert_eq!(
            (
                neuron.v,
                neuron.a_exc,
                neuron.i_exc,
                neuron.a_inh,
                neuron.i_inh
            ),
            before
        );
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        assert!(matches!(
            AlphaNeuron::with_parameters(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 20.0, 5.0, 10.0, 1.0),
            Err(AlphaError::InvalidScale)
        ));
        assert!(matches!(
            AlphaNeuron::with_parameters(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 10.0, 1.0),
            Err(AlphaError::InvalidScale)
        ));
    }

    #[test]
    fn batch_matches_scalar_and_empty_preserves_initial_state() {
        let empty = simulate(
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.0,
            1.0,
            20.0,
            5.0,
            10.0,
            1.0,
            &[],
            &[],
        )
        .unwrap();
        assert!(empty.v.is_empty() && empty.spikes.is_empty());
        assert_eq!(empty.final_state, [0.1, 0.2, 0.3, 0.4, 0.5]);

        let exc = [0.5, 0.6, 0.7, 0.8];
        let inh = [0.1, 0.2, 0.1, 0.2];
        let batch = simulate(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 5.0, 10.0, 1.0, &exc, &inh,
        )
        .unwrap();
        let mut scalar = AlphaNeuron::new();
        let mut count = 0;
        for index in 0..exc.len() {
            count += scalar.try_step(exc[index], inh[index]).unwrap() as usize;
        }
        assert_eq!(
            batch.final_state,
            [
                scalar.v,
                scalar.a_exc,
                scalar.i_exc,
                scalar.a_inh,
                scalar.i_inh
            ]
        );
        assert_eq!(batch.spike_count, count);
    }

    #[test]
    fn reset_restores_documented_rest_state_not_configuration() {
        let mut neuron = AlphaNeuron::new();
        neuron.v = 0.4;
        neuron.a_exc = 0.3;
        neuron.i_inh = 0.2;
        neuron.reset();
        assert_eq!(
            (
                neuron.v,
                neuron.a_exc,
                neuron.i_exc,
                neuron.a_inh,
                neuron.i_inh
            ),
            (0.0, 0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            (
                neuron.v_threshold,
                neuron.tau_v,
                neuron.tau_exc,
                neuron.tau_inh,
                neuron.dt
            ),
            (1.0, 20.0, 5.0, 10.0, 1.0)
        );
    }
}
