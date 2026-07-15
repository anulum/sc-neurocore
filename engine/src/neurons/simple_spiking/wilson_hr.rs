// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wilson HR Neuron Model

//! Wilson HR polynomial cortical neuron dynamics.

/// Wilson HR — polynomial cortical neuron. Wilson 1999.
#[derive(Clone, Debug)]
pub struct WilsonHRNeuron {
    pub v: f64,
    pub r: f64,
    pub tau_r: f64,
    pub v_peak: f64,
    pub dt: f64,
}

impl WilsonHRNeuron {
    pub fn new() -> Self {
        Self {
            v: -0.7,
            r: 0.1,
            tau_r: 1.9,
            v_peak: 0.4,
            dt: 0.05,
        }
    }
    fn valid_numeric_contract(&self) -> bool {
        self.v.is_finite()
            && self.r.is_finite()
            && self.tau_r.is_finite()
            && self.tau_r > 0.0
            && self.v_peak.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
    }
    fn poly(v: f64) -> f64 {
        -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)
    }
    fn derivatives(&self, v: f64, r: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && r.is_finite() && current.is_finite()) {
            return None;
        }
        let poly = Self::poly(v);
        let syn = -26.0 * r * (v + 0.92);
        let dv = poly + syn + current;
        let dr = (-r + 1.35 * v + 1.03) / self.tau_r;
        if poly.is_finite() && syn.is_finite() && dv.is_finite() && dr.is_finite() {
            Some((dv, dr))
        } else {
            None
        }
    }
    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64)> {
        let v0 = self.v;
        let r0 = self.r;
        let dt = self.dt;
        let k1 = self.derivatives(v0, r0, current)?;
        let k2 = self.derivatives(v0 + 0.5 * dt * k1.0, r0 + 0.5 * dt * k1.1, current)?;
        let k3 = self.derivatives(v0 + 0.5 * dt * k2.0, r0 + 0.5 * dt * k2.1, current)?;
        let k4 = self.derivatives(v0 + dt * k3.0, r0 + dt * k3.1, current)?;
        let next_v = v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        let next_r = r0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
        if next_v.is_finite() && next_r.is_finite() {
            Some((next_v, next_r))
        } else {
            None
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return 0;
        }
        let (next_v, next_r) = match self.rk4_candidate(current) {
            Some(candidate) => candidate,
            None => return 0,
        };
        self.v = next_v;
        self.r = next_r;
        // Wilson 1999: hard reset on threshold (not crossing)
        if self.v >= self.v_peak {
            self.v = -0.7;
            1
        } else {
            0
        }
    }
    /// Run `n_steps` RK4 updates under a constant input, returning the `v` trace
    /// (already reset to `-0.7` on spiking steps) and the spike count. Reuses
    /// `step`, so the trace is bit-identical to the per-step path and to the
    /// Python reference (the right-hand side is exact polynomial arithmetic — no
    /// transcendental functions). The final state is left in `self.v` / `self.r`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.v);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }
    pub fn reset(&mut self) {
        self.v = -0.7;
        self.r = 0.1;
    }
}
impl Default for WilsonHRNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = WilsonHRNeuron::default();
        let constructed = WilsonHRNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn simulate_matches_repeated_step() {
        let mut simulated = WilsonHRNeuron::new();
        let mut repeated = WilsonHRNeuron::new();
        let (trace, spikes) = simulated.simulate(2_000, 10.0);
        let mut expected_trace = Vec::with_capacity(2_000);
        let mut expected_spikes = 0_i64;
        for _ in 0..2_000 {
            if repeated.step(10.0) == 1 {
                expected_spikes += 1;
            }
            expected_trace.push(repeated.v);
        }
        assert_eq!(trace, expected_trace);
        assert_eq!(spikes, expected_spikes);
    }

    #[test]
    fn wilson_hr_fires() {
        let mut n = WilsonHRNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn wilson_hr_reset_clears_state() {
        let mut n = WilsonHRNeuron::new();
        for _ in 0..500 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - (-0.7)).abs() < 1e-10);
    }

    #[test]
    fn wilson_hr_moderate_stable() {
        let mut n = WilsonHRNeuron::new();
        for _ in 0..2000 {
            n.step(1.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn wilson_hr_matches_rk4_candidate() {
        fn rhs(n: &WilsonHRNeuron, v: f64, r: f64, current: f64) -> (f64, f64) {
            (
                WilsonHRNeuron::poly(v) - 26.0 * r * (v + 0.92) + current,
                (-r + 1.35 * v + 1.03) / n.tau_r,
            )
        }

        let mut n = WilsonHRNeuron {
            v: -0.4,
            r: 0.08,
            ..Default::default()
        };
        let current = 0.3;
        let v0 = n.v;
        let r0 = n.r;
        let dt = n.dt;
        let k1 = rhs(&n, v0, r0, current);
        let k2 = rhs(&n, v0 + 0.5 * dt * k1.0, r0 + 0.5 * dt * k1.1, current);
        let k3 = rhs(&n, v0 + 0.5 * dt * k2.0, r0 + 0.5 * dt * k2.1, current);
        let k4 = rhs(&n, v0 + dt * k3.0, r0 + dt * k3.1, current);
        let expected_v = v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        let expected_r = r0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected_v).abs() < 1e-14);
        assert!((n.r - expected_r).abs() < 1e-14);
    }

    #[test]
    fn wilson_hr_nan_no_panic() {
        let mut n = WilsonHRNeuron::new();
        let before = (n.v, n.r);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.r), before);
    }

    #[test]
    fn wilson_hr_overflow_candidate_preserves_state() {
        let mut n = WilsonHRNeuron {
            v: 1.0e308,
            ..Default::default()
        };
        let before = (n.v, n.r);
        assert_eq!(n.step(0.3), 0);
        assert_eq!((n.v, n.r), before);
    }

    #[test]
    fn wilson_hr_negative_no_crash() {
        let mut n = WilsonHRNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
}
