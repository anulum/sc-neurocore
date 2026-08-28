// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for wilson_hr

#[derive(Debug, Clone)]
pub struct WilsonHRNeuron {
    pub v: f64,
    pub r: f64,
    pub capacitance: f64,
    pub tau_r: f64,
    pub v_peak: f64,
    pub dt: f64,
}

impl WilsonHRNeuron {
    pub fn new() -> Self {
        Self {
            v: -0.7_f64,
            r: 0.085_f64,
            capacitance: 0.8_f64,
            tau_r: 1.9_f64,
            v_peak: 0.0_f64,
            dt: 0.05_f64,
        }
    }

    fn poly(v: f64) -> f64 {
        -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)
    }

    fn derivatives(&self, v: f64, r: f64, i_ext: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && r.is_finite() && i_ext.is_finite()) {
            return None;
        }
        let poly = Self::poly(v);
        let syn = -26.0 * r * (v + 0.92);
        let dv = (poly + syn + i_ext) / self.capacitance;
        let dr = (-r + 1.35 * v + 1.03) / self.tau_r;
        if poly.is_finite() && syn.is_finite() && dv.is_finite() && dr.is_finite() {
            Some((dv, dr))
        } else {
            None
        }
    }

    fn rk4_candidate(&self, i_ext: f64) -> Option<(f64, f64)> {
        let v0 = self.v;
        let r0 = self.r;
        let dt = self.dt;
        let k1 = self.derivatives(v0, r0, i_ext)?;
        let k2 = self.derivatives(v0 + 0.5 * dt * k1.0, r0 + 0.5 * dt * k1.1, i_ext)?;
        let k3 = self.derivatives(v0 + 0.5 * dt * k2.0, r0 + 0.5 * dt * k2.1, i_ext)?;
        let k4 = self.derivatives(v0 + dt * k3.0, r0 + dt * k3.1, i_ext)?;
        let next_v = v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        let next_r = r0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
        if next_v.is_finite() && next_r.is_finite() {
            Some((next_v, next_r))
        } else {
            None
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_wilson_hr(self) {
            return Err("invalid Wilson-HR runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Wilson-HR external current");
        }
        let (next_v, next_r) = match self.rk4_candidate(i_ext) {
            Some(candidate) => candidate,
            None => return Err("invalid Wilson-HR candidate state"),
        };
        let previous_v = self.v;
        self.v = next_v;
        self.r = next_r;
        Ok(i32::from(self.v >= self.v_peak && previous_v < self.v_peak))
    }

    pub fn simulate(
        &mut self,
        n_steps: usize,
        i_ext: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes = 0_i64;
        for _ in 0..n_steps {
            spikes += i64::from(candidate.step(i_ext)?);
            trace.push(candidate.v);
        }
        *self = candidate;
        Ok((trace, spikes))
    }

    pub fn reset(&mut self) {
        // Restore only state variables; capacitance/tau_r/v_peak/dt are configuration.
        self.v = -0.7_f64;
        self.r = 0.085_f64;
    }
}

impl Default for WilsonHRNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_wilson_hr(state: &WilsonHRNeuron) -> bool {
    state.v.is_finite()
        && state.r.is_finite()
        && state.capacitance.is_finite()
        && state.capacitance > 0.0
        && state.tau_r.is_finite()
        && state.tau_r > 0.0
        && state.v_peak.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rhs(n: &WilsonHRNeuron, v: f64, r: f64, i_ext: f64) -> (f64, f64) {
        (
            (WilsonHRNeuron::poly(v) - 26.0 * r * (v + 0.92) + i_ext) / n.capacitance,
            (-r + 1.35 * v + 1.03) / n.tau_r,
        )
    }

    fn rk4_reference(n: &WilsonHRNeuron, i_ext: f64) -> (f64, f64) {
        let v0 = n.v;
        let r0 = n.r;
        let dt = n.dt;
        let k1 = rhs(n, v0, r0, i_ext);
        let k2 = rhs(n, v0 + 0.5 * dt * k1.0, r0 + 0.5 * dt * k1.1, i_ext);
        let k3 = rhs(n, v0 + 0.5 * dt * k2.0, r0 + 0.5 * dt * k2.1, i_ext);
        let k4 = rhs(n, v0 + dt * k3.0, r0 + dt * k3.1, i_ext);
        (
            v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            r0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
        )
    }

    #[test]
    fn test_wilson_hr_new() {
        let state = WilsonHRNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_wilson_hr(&state));
    }

    #[test]
    fn test_wilson_hr_matches_rk4_candidate() {
        let mut state = WilsonHRNeuron {
            v: -0.4,
            r: 0.08,
            ..WilsonHRNeuron::new()
        };
        let expected = rk4_reference(&state, 0.3);
        assert_eq!(state.step(0.3).unwrap(), 0);
        assert!((state.v - expected.0).abs() < 1e-14);
        assert!((state.r - expected.1).abs() < 1e-14);
    }

    #[test]
    fn test_wilson_hr_step() {
        let mut state = WilsonHRNeuron::new();
        let spike = state.step(0.1).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_wilson_hr_rejects_invalid_runtime_state() {
        let mut state = WilsonHRNeuron::new();
        state.r = f64::INFINITY;
        assert!(state.step(0.3).is_err());
    }

    #[test]
    fn test_wilson_hr_invalid_current_preserves_state() {
        let mut state = WilsonHRNeuron::new();
        let before = (state.v, state.r);
        assert!(state.step(f64::NAN).is_err());
        assert_eq!((state.v, state.r), before);
    }

    #[test]
    fn test_wilson_hr_overflow_candidate_preserves_state() {
        let mut state = WilsonHRNeuron {
            v: 1.0e308,
            ..WilsonHRNeuron::new()
        };
        let before = (state.v, state.r);
        assert!(state.step(0.3).is_err());
        assert_eq!((state.v, state.r), before);
    }

    #[test]
    fn test_wilson_hr_batch_failure_preserves_state() {
        let mut state = WilsonHRNeuron {
            v: 1.0e103,
            ..WilsonHRNeuron::new()
        };
        let before = (state.v, state.r);
        assert!(state.simulate(2, 0.1).is_err());
        assert_eq!((state.v, state.r), before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Source C=0.8 continuous-flow parity with models/wilson_hr.py. Events are
        // sampled upward crossings of v=0 and never reset the trajectory.
        for (current, want) in [(0.0_f64, 0_usize), (0.1, 46), (0.14, 49)] {
            let mut state = WilsonHRNeuron::new();
            let spikes = (0..5000)
                .filter(|_| state.step(current).expect("finite step") == 1)
                .count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
