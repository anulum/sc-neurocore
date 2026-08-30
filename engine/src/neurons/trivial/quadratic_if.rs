// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Latham QIF + preserved symmetric SC profile

/// Quadratic Integrate-and-Fire — canonical Type-I excitability.
/// dv/dt = v² + I, reset at v_peak.
#[derive(Clone, Debug)]
pub struct QuadraticIFNeuron {
    pub v: f64,
    pub v_reset: f64,
    pub v_peak: f64,
    pub dt: f64,
    pub source_profile: bool,
}

pub type QuadraticIFCompleteTrace = (Vec<f64>, Vec<u8>, f64);

impl QuadraticIFNeuron {
    pub fn new(v_reset: f64, v_peak: f64, dt: f64) -> Self {
        Self {
            v: v_reset,
            v_reset,
            v_peak,
            dt,
            source_profile: false,
        }
    }

    /// Construct the normalized numerical profile from Latham et al. (2000).
    pub fn latham_2000() -> Self {
        Self {
            v: -1.0,
            v_reset: -3.0,
            v_peak: 31.0 / 3.0,
            dt: 0.05,
            source_profile: true,
        }
    }

    fn valid_numeric_contract(&self) -> bool {
        self.v.is_finite()
            && self.v_reset.is_finite()
            && self.v_peak.is_finite()
            && self.dt.is_finite()
            && self.v < self.v_peak
            && self.v_reset < self.v_peak
            && self.dt > 0.0
    }

    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return Err("quadratic-IF state/current violates its finite profile contract");
        }
        let (next_v, spiked) = self.exact_candidate(current);
        if !next_v.is_finite() {
            return Err("quadratic-IF exact-flow candidate became non-finite");
        }
        self.v = next_v;
        Ok(i32::from(spiked))
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Execute a complete failure-atomic batch against a cloned state.
    pub fn simulate_complete(
        &self,
        n_steps: usize,
        current: f64,
    ) -> Result<QuadraticIFCompleteTrace, &'static str> {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return Err("invalid quadratic-IF batch contract");
        }
        let mut candidate = self.clone();
        let mut voltage = Vec::with_capacity(n_steps);
        let mut events = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.try_step(current)?;
            voltage.push(candidate.v);
            events.push(event as u8);
        }
        Ok((voltage, events, candidate.v))
    }

    pub fn reset(&mut self) {
        self.v = self.v_reset;
    }

    fn exact_candidate(&self, current: f64) -> (f64, bool) {
        if current > 0.0 {
            let root_i = current.sqrt();
            let phase = (self.v / root_i).atan();
            let peak_phase = (self.v_peak / root_i).atan();
            let next_phase = phase + root_i * self.dt;
            if next_phase >= peak_phase || next_phase >= std::f64::consts::FRAC_PI_2 {
                return (self.v_reset, true);
            }
            return (root_i * next_phase.tan(), false);
        }
        if current == 0.0 {
            let denominator = 1.0 - self.v * self.dt;
            if denominator <= 0.0 {
                return (self.v_reset, true);
            }
            let next_v = self.v / denominator;
            if next_v >= self.v_peak {
                return (self.v_reset, true);
            }
            return (next_v, false);
        }

        let root_i = (-current).sqrt();
        if (self.v + root_i).abs() <= 1e-15 {
            return (self.v, false);
        }
        let numerator_ratio = (self.v - root_i) / (self.v + root_i);
        let evolved_ratio = numerator_ratio * (2.0 * root_i * self.dt).exp();
        let denominator = 1.0 - evolved_ratio;
        if (numerator_ratio < 1.0 && evolved_ratio >= 1.0) || denominator.abs() <= 1e-15 {
            return (self.v_reset, true);
        }
        let next_v = root_i * (1.0 + evolved_ratio) / denominator;
        if next_v >= self.v_peak {
            (self.v_reset, true)
        } else {
            (next_v, false)
        }
    }
}

impl Default for QuadraticIFNeuron {
    fn default() -> Self {
        Self::new(-1.0, 1.0, 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qif_fires_with_positive_input() {
        let mut n = QuadraticIFNeuron::default();
        let total: i32 = (0..1000).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }
    #[test]
    fn qif_silent_without_input() {
        let mut n = QuadraticIFNeuron::default();
        let t: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn qif_reset_clears_state() {
        let mut n = QuadraticIFNeuron::default();
        for _ in 0..100 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - n.v_reset).abs() < 1e-10);
    }
    #[test]
    fn qif_bounded() {
        let mut n = QuadraticIFNeuron::default();
        for _ in 0..1000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn qif_nan_no_panic() {
        let mut n = QuadraticIFNeuron::default();
        let before = n.v;
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!(n.v, before);
    }
    #[test]
    fn latham_profile_has_source_boundaries_and_complete_events() {
        let n = QuadraticIFNeuron::latham_2000();
        assert_eq!(n.v_reset, -3.0);
        assert_eq!(n.v_peak, 31.0 / 3.0);
        assert_eq!(n.dt, 0.05);
        assert!(n.source_profile);
        let (trace, events, final_v) = n.simulate_complete(8, 4.0).unwrap();
        assert_eq!(trace.len(), 8);
        assert_eq!(events.len(), 8);
        assert_eq!(trace.last().copied(), Some(final_v));
    }
    #[test]
    fn complete_batch_rejects_without_mutating_source() {
        let mut n = QuadraticIFNeuron::latham_2000();
        n.v = -0.25;
        assert!(n.simulate_complete(2, f64::NAN).is_err());
        assert_eq!(n.v, -0.25);
    }
    #[test]
    fn qif_nonfinite_increment_preserves_state() {
        let mut n = QuadraticIFNeuron {
            v: -0.25,
            ..Default::default()
        };
        let before = n.v;
        assert_eq!(n.step(-1.0e308), 0);
        assert_eq!(n.v, before);
    }
    #[test]
    fn qif_matches_exact_positive_current_flow() {
        let mut n = QuadraticIFNeuron::default();
        let root_i = 0.5_f64.sqrt();
        let expected = root_i * ((n.v / root_i).atan() + root_i * n.dt).tan();
        assert_eq!(n.step(0.5), 0);
        assert!((n.v - expected).abs() < 1e-12);
    }
    #[test]
    fn qif_preserves_negative_current_fixed_point() {
        let mut n = QuadraticIFNeuron::default();
        assert_eq!(n.step(-1.0), 0);
        assert_eq!(n.v, -1.0);
    }
    #[test]
    fn qif_exact_flow_resets_on_peak_crossing() {
        let mut n = QuadraticIFNeuron {
            v: 0.95,
            dt: 0.5,
            ..Default::default()
        };
        assert_eq!(n.step(1.0), 1);
        assert_eq!(n.v, n.v_reset);
    }
    #[test]
    fn qif_negative_no_crash() {
        let mut n = QuadraticIFNeuron::default();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
}
