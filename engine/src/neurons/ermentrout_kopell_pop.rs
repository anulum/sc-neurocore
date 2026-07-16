// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Montbrió-Pazó-Roxin exact QIF mean-field engine

//! The source prints dimensionless equations (12a-b) in `(R, v, t')`.
//! This engine restores physical rate and time through `R = tau * r` and
//! `t' = t / tau`, then applies atomic simultaneous-Euler batches.

use std::error::Error;
use std::fmt;

/// Typed numerical and caller-contract failures for the MPR population.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErmentroutKopellPopulationError {
    /// One state or configuration value is not finite.
    NonFiniteConfiguration,
    /// The initial population firing rate is negative.
    NegativeInitialRate,
    /// The time scale or explicit-Euler step is not positive.
    NonPositiveTimeScale,
    /// The Lorentzian half-width is negative.
    NegativeHalfWidth,
    /// One external-drive value is not finite.
    NonFiniteInput,
    /// A simultaneous Euler candidate contains a non-finite state.
    NonFiniteCandidate,
    /// A simultaneous Euler candidate has a negative firing rate.
    NegativeCandidateRate,
}

impl fmt::Display for ErmentroutKopellPopulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFiniteConfiguration => "MPR state and parameters must be finite",
            Self::NegativeInitialRate => "MPR firing rate must be non-negative",
            Self::NonPositiveTimeScale => "MPR tau and dt must be positive",
            Self::NegativeHalfWidth => "MPR Lorentzian half-width must be non-negative",
            Self::NonFiniteInput => "MPR external input must contain only finite values",
            Self::NonFiniteCandidate => "MPR candidate state must remain finite",
            Self::NegativeCandidateRate => "MPR candidate firing rate became negative",
        };
        formatter.write_str(message)
    }
}

impl Error for ErmentroutKopellPopulationError {}

/// Legacy-named public wrapper for the Montbrió-Pazó-Roxin population model.
#[derive(Clone, Debug)]
pub struct ErmentroutKopellPopulation {
    /// Population firing rate.
    pub r: f64,
    /// Mean membrane potential.
    pub v: f64,
    /// Positive membrane time scale.
    pub tau: f64,
    /// Non-negative Lorentzian half-width of neuronal excitability.
    pub delta: f64,
    /// Centre of the neuronal excitability distribution.
    pub eta_bar: f64,
    /// Recurrent coupling strength.
    pub j: f64,
    /// Positive explicit-Euler step.
    pub dt: f64,
}

impl ErmentroutKopellPopulation {
    /// Construct the phase-portrait parameter set used by the source paper.
    pub fn new() -> Self {
        Self {
            r: 0.1,
            v: -2.0,
            tau: 1.0,
            delta: 1.0,
            eta_bar: -5.0,
            j: 15.0,
            dt: 0.01,
        }
    }

    /// Construct and validate one complete numerical configuration.
    pub fn with_parameters(
        r: f64,
        v: f64,
        tau: f64,
        delta: f64,
        eta_bar: f64,
        j: f64,
        dt: f64,
    ) -> Result<Self, ErmentroutKopellPopulationError> {
        let population = Self {
            r,
            v,
            tau,
            delta,
            eta_bar,
            j,
            dt,
        };
        population.validate()?;
        Ok(population)
    }

    fn validate(&self) -> Result<(), ErmentroutKopellPopulationError> {
        if ![
            self.r,
            self.v,
            self.tau,
            self.delta,
            self.eta_bar,
            self.j,
            self.dt,
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            return Err(ErmentroutKopellPopulationError::NonFiniteConfiguration);
        }
        if self.r < 0.0 {
            return Err(ErmentroutKopellPopulationError::NegativeInitialRate);
        }
        if self.tau <= 0.0 || self.dt <= 0.0 {
            return Err(ErmentroutKopellPopulationError::NonPositiveTimeScale);
        }
        if self.delta < 0.0 {
            return Err(ErmentroutKopellPopulationError::NegativeHalfWidth);
        }
        Ok(())
    }

    #[inline]
    fn derivatives(&self, drive: f64) -> (f64, f64) {
        let scaled_rate = std::f64::consts::PI * self.tau * self.r;
        let dr = self.delta / (std::f64::consts::PI * self.tau * self.tau)
            + 2.0 * self.r * self.v / self.tau;
        let dv = (self.v * self.v + self.eta_bar + drive + self.j * self.tau * self.r
            - scaled_rate * scaled_rate)
            / self.tau;
        (dr, dv)
    }

    /// Advance one simultaneous explicit-Euler step with explicit error reporting.
    pub fn try_step(&mut self, ext_input: f64) -> Result<f64, ErmentroutKopellPopulationError> {
        self.validate()?;
        if !ext_input.is_finite() {
            return Err(ErmentroutKopellPopulationError::NonFiniteInput);
        }
        let (dr, dv) = self.derivatives(ext_input);
        let next_r = self.r + self.dt * dr;
        let next_v = self.v + self.dt * dv;
        if !next_r.is_finite() || !next_v.is_finite() {
            return Err(ErmentroutKopellPopulationError::NonFiniteCandidate);
        }
        if next_r < 0.0 {
            return Err(ErmentroutKopellPopulationError::NegativeCandidateRate);
        }
        self.r = next_r;
        self.v = next_v;
        Ok(self.r)
    }

    /// Advance one step while preserving the legacy scalar-returning Rust API.
    ///
    /// Invalid input is fail-closed: the state remains unchanged and the
    /// current firing rate is returned. New callers that need diagnostics
    /// should use [`Self::try_step`].
    pub fn step(&mut self, ext_input: f64) -> f64 {
        match self.try_step(ext_input) {
            Ok(rate) => rate,
            Err(_) => self.r,
        }
    }

    /// Restore dynamic states while preserving the configured parameters.
    pub fn reset(&mut self) {
        self.r = 0.1;
        self.v = -2.0;
    }
}

impl Default for ErmentroutKopellPopulation {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-step states and final receipt from one complete drive batch.
pub struct ErmentroutKopellPopulationTrace {
    /// Post-update firing-rate trace.
    pub r: Vec<f64>,
    /// Post-update mean-voltage trace.
    pub v: Vec<f64>,
    /// Final ``[r, v]`` receipt, including the initial state for an empty batch.
    pub final_state: [f64; 2],
}

/// Simulate a complete caller-owned external-drive vector atomically.
#[expect(
    clippy::too_many_arguments,
    reason = "native parity surface carries the complete scientific configuration"
)]
pub fn simulate(
    r: f64,
    v: f64,
    tau: f64,
    delta: f64,
    eta_bar: f64,
    j: f64,
    dt: f64,
    ext_input: &[f64],
) -> Result<ErmentroutKopellPopulationTrace, ErmentroutKopellPopulationError> {
    let mut population =
        ErmentroutKopellPopulation::with_parameters(r, v, tau, delta, eta_bar, j, dt)?;
    if !ext_input.iter().all(|value| value.is_finite()) {
        return Err(ErmentroutKopellPopulationError::NonFiniteInput);
    }
    let mut r_trace = Vec::with_capacity(ext_input.len());
    let mut v_trace = Vec::with_capacity(ext_input.len());
    for drive in ext_input {
        population.try_step(*drive)?;
        r_trace.push(population.r);
        v_trace.push(population.v);
    }
    Ok(ErmentroutKopellPopulationTrace {
        r: r_trace,
        v: v_trace,
        final_state: [population.r, population.v],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_step_matches_equation_twelve_with_explicit_tau() {
        let mut population =
            ErmentroutKopellPopulation::with_parameters(0.2, -1.5, 2.0, 0.7, -3.0, 12.0, 0.005)
                .unwrap();
        let old_r = population.r;
        let old_v = population.v;
        let drive = 1.25;
        let expected_r =
            old_r + 0.005 * (0.7 / (std::f64::consts::PI * 4.0) + 2.0 * old_r * old_v / 2.0);
        let expected_v = old_v
            + 0.005
                * (old_v * old_v + -3.0 + drive + 12.0 * 2.0 * old_r
                    - (std::f64::consts::PI * 2.0 * old_r).powi(2))
                / 2.0;
        population.try_step(drive).unwrap();
        assert_eq!(population.r, expected_r);
        assert_eq!(population.v, expected_v);
    }

    #[test]
    fn invalid_step_is_atomic() {
        let mut population = ErmentroutKopellPopulation::new();
        let before = (population.r, population.v);
        assert_eq!(
            population.try_step(f64::NAN),
            Err(ErmentroutKopellPopulationError::NonFiniteInput)
        );
        assert_eq!((population.r, population.v), before);
    }

    #[test]
    fn candidate_failures_are_typed_and_atomic() {
        let mut negative =
            ErmentroutKopellPopulation::with_parameters(1.0, -100.0, 1.0, 0.0, 0.0, 0.0, 0.1)
                .unwrap();
        let before = (negative.r, negative.v);
        assert_eq!(
            negative.try_step(0.0),
            Err(ErmentroutKopellPopulationError::NegativeCandidateRate)
        );
        assert_eq!((negative.r, negative.v), before);

        let mut nonfinite = ErmentroutKopellPopulation::with_parameters(
            f64::MAX,
            f64::MAX,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        .unwrap();
        let before = (nonfinite.r, nonfinite.v);
        assert_eq!(
            nonfinite.try_step(0.0),
            Err(ErmentroutKopellPopulationError::NonFiniteCandidate)
        );
        assert_eq!((nonfinite.r, nonfinite.v), before);
    }

    #[test]
    fn batch_matches_scalar_and_empty_preserves_initial_state() {
        let empty = simulate(0.2, -1.5, 2.0, 0.7, -3.0, 12.0, 0.005, &[]).unwrap();
        assert!(empty.r.is_empty() && empty.v.is_empty());
        assert_eq!(empty.final_state, [0.2, -1.5]);

        let drive = [0.0, 0.25, -0.1, 1.0];
        let batch = simulate(0.1, -2.0, 1.0, 1.0, -5.0, 15.0, 0.01, &drive).unwrap();
        let mut scalar = ErmentroutKopellPopulation::new();
        for value in drive {
            scalar.try_step(value).unwrap();
        }
        assert_eq!(batch.final_state, [scalar.r, scalar.v]);
    }
}
