// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hindmarsh-Rose Neuron Model

//! Hindmarsh-Rose bursting neuron dynamics.

/// Hindmarsh-Rose 1984 — 3D chaotic bursting model.
#[derive(Clone, Debug)]
pub struct HindmarshRoseNeuron {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub b: f64,
    pub r: f64,
    pub s: f64,
    pub x_rest: f64,
    pub dt: f64,
    pub x_threshold: f64,
}

impl HindmarshRoseNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.6,
            y: -10.0,
            z: 2.0,
            b: 3.0,
            r: 0.001,
            s: 4.0,
            x_rest: -1.6,
            dt: 0.1,
            x_threshold: 1.0,
        }
    }
    fn valid_state(&self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.z.is_finite()
            && self.b.is_finite()
            && self.r.is_finite()
            && self.s.is_finite()
            && self.x_rest.is_finite()
            && self.dt.is_finite()
            && self.x_threshold.is_finite()
            && self.r > 0.0
            && self.s > 0.0
            && self.dt > 0.0
    }
    fn derivatives(&self, x: f64, y: f64, z: f64, current: f64) -> Option<(f64, f64, f64)> {
        if !(x.is_finite() && y.is_finite() && z.is_finite() && current.is_finite()) {
            return None;
        }
        let derivative = (
            y - x.powi(3) + self.b * x.powi(2) - z + current,
            1.0 - 5.0 * x.powi(2) - y,
            self.r * (self.s * (x - self.x_rest) - z),
        );
        if derivative.0.is_finite() && derivative.1.is_finite() && derivative.2.is_finite() {
            Some(derivative)
        } else {
            None
        }
    }
    fn try_step(&mut self, current: f64) -> Option<i32> {
        if !self.valid_state() || !current.is_finite() {
            return None;
        }
        let x_prev = self.x;
        let (x0, y0, z0) = (self.x, self.y, self.z);
        let dt = self.dt;
        let k1 = self.derivatives(x0, y0, z0, current)?;
        let k2 = self.derivatives(
            x0 + 0.5 * dt * k1.0,
            y0 + 0.5 * dt * k1.1,
            z0 + 0.5 * dt * k1.2,
            current,
        )?;
        let k3 = self.derivatives(
            x0 + 0.5 * dt * k2.0,
            y0 + 0.5 * dt * k2.1,
            z0 + 0.5 * dt * k2.2,
            current,
        )?;
        let k4 = self.derivatives(x0 + dt * k3.0, y0 + dt * k3.1, z0 + dt * k3.2, current)?;
        let next_x = x0 + (dt / 6.0) * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
        let next_y = y0 + (dt / 6.0) * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
        let next_z = z0 + (dt / 6.0) * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2);
        if !(next_x.is_finite() && next_y.is_finite() && next_z.is_finite()) {
            return None;
        }
        self.x = next_x;
        self.y = next_y;
        self.z = next_z;
        Some(if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        })
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }
    pub fn reset(&mut self) {
        self.x = -1.6;
        self.y = -10.0;
        self.z = 2.0;
    }

    /// Run `n_steps` RK4 updates under a constant input, returning the `x` trace
    /// and the upward-crossing spike count. Reuses `step` (RK4) so the trace is
    /// bit-identical to the per-step path and — because the right-hand side is
    /// exact arithmetic (`x.powi(3)` = `x*x*x`, `x.powi(2)` = `x*x`) — to the
    /// Python reference, even though the bursting dynamics are chaotic. The
    /// final state is left in `self.x` / `self.y` / `self.z`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        self.try_simulate(n_steps, current).unwrap_or_default()
    }

    /// Run one failure-atomic batch, returning `None` on any invalid stage.
    pub fn try_simulate(&mut self, n_steps: usize, current: f64) -> Option<(Vec<f64>, i64)> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = candidate.try_step(current)?;
            trace.push(candidate.x);
            if spiked == 1 {
                spikes += 1;
            }
        }
        *self = candidate;
        Some((trace, spikes))
    }
}
impl Default for HindmarshRoseNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = HindmarshRoseNeuron::default();
        let constructed = HindmarshRoseNeuron::new();
        assert_eq!(default.x, constructed.x);
    }

    #[test]
    fn simulate_matches_repeated_step() {
        let mut simulated = HindmarshRoseNeuron::new();
        let mut repeated = HindmarshRoseNeuron::new();
        let (trace, spikes) = simulated.simulate(2_000, 3.0);
        let mut expected_trace = Vec::with_capacity(2_000);
        let mut expected_spikes = 0_i64;
        for _ in 0..2_000 {
            if repeated.step(3.0) == 1 {
                expected_spikes += 1;
            }
            expected_trace.push(repeated.x);
        }
        assert_eq!(trace, expected_trace);
        assert_eq!(spikes, expected_spikes);
    }

    #[test]
    fn try_simulate_rejects_overflow_without_mutation() {
        let mut neuron = HindmarshRoseNeuron::new();
        neuron.x = 1.0e103;
        let before = (neuron.x, neuron.y, neuron.z);
        assert!(neuron.try_simulate(2, 0.0).is_none());
        assert_eq!((neuron.x, neuron.y, neuron.z), before);
    }

    #[test]
    fn hr_fires() {
        let mut n = HindmarshRoseNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(3.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn hr_reset_clears_state() {
        let mut n = HindmarshRoseNeuron::new();
        for _ in 0..500 {
            n.step(3.0);
        }
        n.reset();
        assert!((n.x - (-1.6)).abs() < 1e-10);
    }

    #[test]
    fn hr_moderate_input_stable() {
        let mut n = HindmarshRoseNeuron::new();
        for _ in 0..2000 {
            n.step(5.0);
        }
        assert!(n.x.is_finite());
    }

    #[test]
    fn hr_slow_z_evolves() {
        let mut n = HindmarshRoseNeuron::new();
        let z0 = n.z;
        for _ in 0..2000 {
            n.step(3.0);
        }
        assert!((n.z - z0).abs() > 0.001, "slow variable z should evolve");
    }

    #[test]
    fn hr_nan_no_panic() {
        HindmarshRoseNeuron::new().step(f64::NAN);
    }

    #[test]
    fn hr_negative_no_crash() {
        let mut n = HindmarshRoseNeuron::new();
        for _ in 0..500 {
            n.step(-1.0);
        }
        assert!(n.x.is_finite());
    }
}
