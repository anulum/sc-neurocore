// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pernarowski Neuron Model

//! Pernarowski beta-cell bursting dynamics.

/// Pernarowski 1994 — simplified beta cell burster (3 ODE).
#[derive(Clone, Debug)]
pub struct PernarowskiNeuron {
    pub v: f64,
    pub w: f64,
    pub z: f64,
    pub alpha: f64,
    pub beta: f64,
    pub eps1: f64,
    pub eps2: f64,
    pub gamma: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PernarowskiNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0,
            w: 0.0,
            z: 0.0,
            alpha: 0.1,
            beta: 0.5,
            eps1: 0.1,
            eps2: 0.001,
            gamma: 0.5,
            dt: 0.1,
            v_threshold: 0.5,
        }
    }
    fn is_valid(&self) -> bool {
        self.v.is_finite()
            && self.w.is_finite()
            && self.z.is_finite()
            && self.alpha.is_finite()
            && self.beta.is_finite()
            && self.eps1.is_finite()
            && self.eps1 > 0.0
            && self.eps2.is_finite()
            && self.eps2 > 0.0
            && self.gamma.is_finite()
            && self.gamma > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }
    fn derivatives(&self, v: f64, w: f64, z: f64, current: f64) -> Option<(f64, f64, f64)> {
        if !(v.is_finite() && w.is_finite() && z.is_finite() && current.is_finite()) {
            return None;
        }
        let dv = v - v.powi(3) / 3.0 - w - z + current;
        let dw = self.eps1 * (v - self.gamma * w + self.alpha);
        let dz = self.eps2 * (self.beta * (v + 0.7) - z);
        if dv.is_finite() && dw.is_finite() && dz.is_finite() {
            Some((dv, dw, dz))
        } else {
            None
        }
    }
    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64, f64)> {
        let dt = self.dt;
        let (k1v, k1w, k1z) = self.derivatives(self.v, self.w, self.z, current)?;
        let (k2v, k2w, k2z) = self.derivatives(
            self.v + 0.5 * dt * k1v,
            self.w + 0.5 * dt * k1w,
            self.z + 0.5 * dt * k1z,
            current,
        )?;
        let (k3v, k3w, k3z) = self.derivatives(
            self.v + 0.5 * dt * k2v,
            self.w + 0.5 * dt * k2w,
            self.z + 0.5 * dt * k2z,
            current,
        )?;
        let (k4v, k4w, k4z) = self.derivatives(
            self.v + dt * k3v,
            self.w + dt * k3w,
            self.z + dt * k3z,
            current,
        )?;
        let v = self.v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;
        let w = self.w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0;
        let z = self.z + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0;
        if v.is_finite() && w.is_finite() && z.is_finite() {
            Some((v, w, z))
        } else {
            None
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        if !self.is_valid() || !current.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let Some((v, w, z)) = self.rk4_candidate(current) else {
            return 0;
        };
        self.v = v;
        self.w = w;
        self.z = z;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    /// Run `n_steps` RK4 updates under a constant input, returning the `v` trace
    /// and the upward-`v_threshold`-crossing spike count. Reuses `step`, so the
    /// trace is bit-identical to the per-step path and to the Python reference
    /// (the cubic uses `v.powi(3)` = `v*v*v`, matching the Python `v*v*v`; no
    /// transcendental functions). The final state is left in `self.v` / `self.w`
    /// / `self.z`.
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
        self.v = -1.0;
        self.w = 0.0;
        self.z = 0.0;
    }
}
impl Default for PernarowskiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = PernarowskiNeuron::default();
        let constructed = PernarowskiNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn simulate_matches_repeated_step() {
        let mut simulated = PernarowskiNeuron::new();
        let mut repeated = PernarowskiNeuron::new();
        let (trace, spikes) = simulated.simulate(2_000, 1.0);
        let mut expected_trace = Vec::with_capacity(2_000);
        let mut expected_spikes = 0_i64;
        for _ in 0..2_000 {
            if repeated.step(1.0) == 1 {
                expected_spikes += 1;
            }
            expected_trace.push(repeated.v);
        }
        assert_eq!(trace, expected_trace);
        assert_eq!(spikes, expected_spikes);
    }

    #[test]
    fn pernarowski_fires() {
        let mut n = PernarowskiNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn pernarowski_reset_clears_state() {
        let mut n = PernarowskiNeuron::new();
        for _ in 0..500 {
            n.step(1.0);
        }
        n.reset();
        assert!((n.v - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn pernarowski_bounded() {
        let mut n = PernarowskiNeuron::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn pernarowski_slow_z() {
        let mut n = PernarowskiNeuron::new();
        let z0 = n.z;
        for _ in 0..2000 {
            n.step(1.0);
        }
        assert!((n.z - z0).abs() > 1e-6, "slow z should evolve");
    }

    #[test]
    fn pernarowski_matches_rk4_candidate() {
        let mut n = PernarowskiNeuron::new();
        n.v = -0.8;
        n.w = 0.2;
        n.z = -0.1;
        let current = 0.5;
        let dt = n.dt;
        let rhs = |v: f64, w: f64, z: f64| {
            (
                v - v.powi(3) / 3.0 - w - z + current,
                n.eps1 * (v - n.gamma * w + n.alpha),
                n.eps2 * (n.beta * (v + 0.7) - z),
            )
        };
        let (k1v, k1w, k1z) = rhs(n.v, n.w, n.z);
        let (k2v, k2w, k2z) = rhs(
            n.v + 0.5 * dt * k1v,
            n.w + 0.5 * dt * k1w,
            n.z + 0.5 * dt * k1z,
        );
        let (k3v, k3w, k3z) = rhs(
            n.v + 0.5 * dt * k2v,
            n.w + 0.5 * dt * k2w,
            n.z + 0.5 * dt * k2z,
        );
        let (k4v, k4w, k4z) = rhs(n.v + dt * k3v, n.w + dt * k3w, n.z + dt * k3z);
        let expected = (
            n.v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
            n.w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
            n.z + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0,
        );

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected.0).abs() < 1e-14);
        assert!((n.w - expected.1).abs() < 1e-14);
        assert!((n.z - expected.2).abs() < 1e-14);
    }

    #[test]
    fn pernarowski_invalid_input_preserves_state() {
        let mut n = PernarowskiNeuron::new();
        let before = (n.v, n.w, n.z);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.w, n.z), before);
    }

    #[test]
    fn pernarowski_overflow_candidate_preserves_state() {
        let mut n = PernarowskiNeuron::new();
        n.v = 1.0e160;
        let before = (n.v, n.w, n.z);
        assert_eq!(n.step(0.5), 0);
        assert_eq!((n.v, n.w, n.z), before);
    }

    #[test]
    fn pernarowski_nan_no_panic() {
        PernarowskiNeuron::new().step(f64::NAN);
    }

    #[test]
    fn pernarowski_negative_no_crash() {
        let mut n = PernarowskiNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
}
