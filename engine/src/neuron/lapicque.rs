// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Lapicque 1907 polarization + preserved SC hard-reset LIF

#![deny(missing_docs)]

mod contract;

/// Profile-explicit Lapicque state used by the production engine.
#[derive(Clone, Debug)]
pub struct LapicqueNeuron {
    /// Membrane (polarisation) potential, normalised.
    pub v: f64,
    /// Resting potential of the SC profile; the source profile does not read it.
    pub v_rest: f64,
    /// Value after a spike in the SC profile; the source profile never resets.
    pub v_reset: f64,
    /// Excitation polarisation threshold (normalised polarisation).
    pub v_threshold: f64,
    /// Membrane time constant of the SC profile. The source profile uses
    /// `K * R * rho / (R + rho)` instead.
    pub tau: f64,
    /// Input resistance of the SC profile. The source profile uses `rho / (R + rho)`.
    pub resistance: f64,
    /// Step; the update is exact for a drive held over `dt`.
    pub dt: f64,
    /// Lapicque circuit capacitance K (normalised capacitance).
    pub capacitance: f64,
    /// Lapicque source-series resistance R (normalised resistance).
    pub series_resistance: f64,
    /// Lapicque polarisation leakage resistance rho (normalised resistance).
    pub polarization_resistance: f64,
    /// Latched first threshold attainment.
    pub excited: bool,
    /// `true` selects the one-shot Lapicque 1907 circuit (latched event, no reset); `false`
    /// the SC hard-reset LIF.
    pub source_profile: bool,
}

/// Complete failure-atomic Lapicque batch result.
pub type LapicqueCompleteTrace = (Vec<f64>, Vec<u8>, f64, bool);

impl LapicqueNeuron {
    /// Construct the preserved SC exact-flow, hard-reset LIF profile.
    pub fn new(tau: f64, resistance: f64, threshold: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            v_threshold: threshold,
            tau,
            resistance,
            dt,
            capacitance: 1.1,
            series_resistance: 10.0,
            polarization_resistance: 1.0,
            excited: false,
            source_profile: false,
        }
    }

    /// Construct the normalized, one-shot Lapicque 1907 source profile.
    pub fn lapicque_1907() -> Self {
        let mut state = Self::new(20.0, 1.0, 1.0, 0.01);
        state.source_profile = true;
        state
    }

    /// Advance one exact constant-drive step without conflating errors with silence.
    ///
    /// `v` moves to `v_inf + (v - v_inf) * exp(-dt / time_constant)`. The SC profile uses
    /// `v_inf = v_rest + resistance * drive` and `tau`; reaching `v_threshold` sets `v`
    /// to `v_reset` and returns 1. The source profile uses
    /// `v_inf = drive * rho / (R + rho)` and `K * R * rho / (R + rho)`; its first reach of
    /// `v_threshold` latches `excited` and returns 1, and `v` is never reset. An error
    /// leaves the state unchanged.
    pub fn try_step(&mut self, drive: f64) -> Result<i32, &'static str> {
        if !drive.is_finite() {
            return Err("Lapicque drive must be finite");
        }
        if !self.valid() {
            return Err("Lapicque state violates its profile contract");
        }

        let (v_inf, decay) = if self.source_profile {
            let total_resistance = self.series_resistance + self.polarization_resistance;
            let beta = self.capacitance * self.series_resistance * self.polarization_resistance
                / total_resistance;
            (
                drive * self.polarization_resistance / total_resistance,
                (-self.dt / beta).exp(),
            )
        } else {
            (
                self.v_rest + self.resistance * drive,
                (-self.dt / self.tau).exp(),
            )
        };
        let next_v = v_inf + (self.v - v_inf) * decay;
        if !v_inf.is_finite() || !decay.is_finite() || !next_v.is_finite() {
            return Err("Lapicque candidate must remain finite");
        }

        if self.source_profile {
            let event = !self.excited && next_v >= self.v_threshold;
            self.v = next_v;
            if event {
                self.excited = true;
                return Ok(1);
            }
            return Ok(0);
        }

        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            Ok(1)
        } else {
            self.v = next_v;
            Ok(0)
        }
    }

    /// Compatibility dispatch for NetworkRunner's uniform non-throwing trait; a refused
    /// step returns 0 and leaves the state unchanged.
    pub fn step(&mut self, drive: f64) -> i32 {
        self.try_step(drive).unwrap_or(0)
    }

    /// Execute a failure-atomic complete batch against a cloned candidate.
    ///
    /// The receiver is never changed. The result holds the post-step `v` and event of
    /// each step, then the final `v` and `excited`.
    pub fn simulate_complete(
        &self,
        n_steps: usize,
        drive: f64,
    ) -> Result<LapicqueCompleteTrace, &'static str> {
        if !drive.is_finite() || !self.valid() {
            return Err("invalid Lapicque batch contract");
        }
        let mut candidate = self.clone();
        let mut voltage = Vec::with_capacity(n_steps);
        let mut events = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.try_step(drive)?;
            voltage.push(candidate.v);
            events.push(event as u8);
        }
        Ok((voltage, events, candidate.v, candidate.excited))
    }

    /// Re-arm a source experiment or restore the SC membrane to rest.
    pub fn reset(&mut self) {
        self.v = if self.source_profile {
            0.0
        } else {
            self.v_rest
        };
        self.excited = false;
    }
}

#[cfg(test)]
#[path = "lapicque_tests.rs"]
mod tests;
