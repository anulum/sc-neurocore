// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Lapicque 1907 polarization + preserved SC hard-reset LIF

/// Profile-explicit Lapicque state used by the production engine.
#[derive(Clone, Debug)]
pub struct LapicqueNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub resistance: f64,
    pub dt: f64,
    pub capacitance: f64,
    pub series_resistance: f64,
    pub polarization_resistance: f64,
    pub excited: bool,
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

    /// Return whether all configuration and dynamic invariants hold.
    pub fn valid(&self) -> bool {
        if !self.v.is_finite()
            || !self.v_threshold.is_finite()
            || self.v_threshold <= 0.0
            || !self.dt.is_finite()
            || self.dt <= 0.0
        {
            return false;
        }
        if self.source_profile {
            return (self.excited || self.v < self.v_threshold)
                && self.capacitance.is_finite()
                && self.capacitance > 0.0
                && self.series_resistance.is_finite()
                && self.series_resistance > 0.0
                && self.polarization_resistance.is_finite()
                && self.polarization_resistance > 0.0;
        }
        !self.excited
            && self.v_rest.is_finite()
            && self.v_reset.is_finite()
            && self.v_threshold > self.v_rest
            && self.v_threshold > self.v_reset
            && self.v < self.v_threshold
            && self.tau.is_finite()
            && self.tau > 0.0
            && self.resistance.is_finite()
            && self.resistance > 0.0
    }

    /// Advance one exact constant-drive step without conflating errors with silence.
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

    /// Compatibility dispatch for NetworkRunner's uniform non-throwing trait.
    pub fn step(&mut self, drive: f64) -> i32 {
        self.try_step(drive).unwrap_or(0)
    }

    /// Execute a failure-atomic complete batch against a cloned candidate.
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
