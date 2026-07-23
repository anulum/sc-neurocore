// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — NeuroGrid Neuron Emulator

/// NeuroGrid — Boahen 2014 subthreshold analog 2-compartment.
#[derive(Clone, Debug)]
pub struct NeuroGridNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub tau_s: f64,
    pub tau_d: f64,
    pub g_c: f64,
    pub delta_t: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub v_peak: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl NeuroGridNeuron {
    pub fn new() -> Self {
        Self {
            v_s: -65.0,
            v_d: -65.0,
            tau_s: 20.0,
            tau_d: 50.0,
            g_c: 0.5,
            delta_t: 2.0,
            v_rest: -65.0,
            v_threshold: -50.0,
            v_peak: 20.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }
    fn valid(&self) -> bool {
        self.v_s.is_finite()
            && self.v_d.is_finite()
            && self.tau_s.is_finite()
            && self.tau_s > 0.0
            && self.tau_d.is_finite()
            && self.tau_d > 0.0
            && self.g_c.is_finite()
            && self.g_c >= 0.0
            && self.delta_t.is_finite()
            && self.delta_t > 0.0
            && self.v_rest.is_finite()
            && self.v_threshold.is_finite()
            && self.v_peak.is_finite()
            && self.v_reset.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
    }

    fn derivatives(&self, v_s: f64, v_d: f64, current: f64) -> (f64, f64) {
        let v_s_eff = v_s.min(self.v_peak);
        let dv_d = (-(v_d - self.v_rest) + current - self.g_c * (v_d - v_s_eff)) / self.tau_d;
        let exp_arg = ((v_s_eff - self.v_threshold) / self.delta_t).min(20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv_s = (-(v_s_eff - self.v_rest) + exp_term + self.g_c * (v_d - v_s_eff)) / self.tau_s;
        (dv_s, dv_d)
    }

    fn rk4_substep(&self, v_s: f64, v_d: f64, current: f64) -> (f64, f64) {
        let dt = self.dt;
        let (k1s, k1d) = self.derivatives(v_s, v_d, current);
        let (k2s, k2d) = self.derivatives(v_s + 0.5 * dt * k1s, v_d + 0.5 * dt * k1d, current);
        let (k3s, k3d) = self.derivatives(v_s + 0.5 * dt * k2s, v_d + 0.5 * dt * k2d, current);
        let (k4s, k4d) = self.derivatives(v_s + dt * k3s, v_d + dt * k3d, current);
        (
            v_s + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0,
            v_d + dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0,
        )
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid() {
            return 0;
        }
        let (next_v_s, next_v_d) = self.rk4_substep(self.v_s, self.v_d, current);
        if !next_v_s.is_finite() || !next_v_d.is_finite() {
            return 0;
        }
        self.v_d = next_v_d;
        if next_v_s >= self.v_peak {
            self.v_s = self.v_reset;
            1
        } else {
            self.v_s = next_v_s;
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = -65.0;
        self.v_d = -65.0;
    }
}
impl Default for NeuroGridNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neurogrid_fires() {
        let mut n = NeuroGridNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(500.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn neurogrid_silent() {
        let mut n = NeuroGridNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn neurogrid_reset() {
        let mut n = NeuroGridNeuron::new();
        for _ in 0..100 {
            n.step(500.0);
        }
        n.reset();
        assert!((n.v_s - (-65.0)).abs() < 1e-10);
    }
    #[test]
    fn neurogrid_bounded() {
        let mut n = NeuroGridNeuron::new();
        for _ in 0..2000 {
            n.step(1e4);
        }
        assert!(n.v_s.is_finite());
    }
    #[test]
    fn neurogrid_nan_no_panic() {
        NeuroGridNeuron::new().step(f64::NAN);
    }
    #[test]
    fn neurogrid_rk4_anchor() {
        let mut n = NeuroGridNeuron::new();
        let spikes: i32 = (0..20_000).map(|_| n.step(100.0)).sum();
        assert_eq!(spikes, 94);
        assert!(n.v_s.is_finite());
        assert!(n.v_d.is_finite());
    }
    #[test]
    fn neurogrid_invalid_input_preserves_state() {
        let mut n = NeuroGridNeuron::new();
        for _ in 0..10 {
            n.step(100.0);
        }
        let old = (n.v_s, n.v_d);
        assert_eq!(n.step(f64::INFINITY), 0);
        assert_eq!((n.v_s, n.v_d), old);
    }
}
