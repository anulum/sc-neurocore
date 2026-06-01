// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for Prescott 2008 RK4 dynamics

#[derive(Debug, Clone)]
pub struct PrescottNeuron {
    pub v: f64,
    pub w: f64,
    pub g_fast: f64,
    pub g_slow: f64,
    pub g_l: f64,
    pub e_fast: f64,
    pub e_slow: f64,
    pub e_l: f64,
    pub beta_w: f64,
    pub gamma_w: f64,
    pub tau_w: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PrescottNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            g_fast: 20.0,
            g_slow: 20.0,
            g_l: 2.0,
            e_fast: 50.0,
            e_slow: -100.0,
            e_l: -70.0,
            beta_w: -21.0,
            gamma_w: 15.0,
            tau_w: 100.0,
            phi: 0.15,
            dt: 0.1,
            v_threshold: -20.0,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }

    fn valid_state(v: f64, w: f64) -> bool {
        v.is_finite() && w.is_finite() && (0.0..=1.0).contains(&w)
    }

    fn valid_runtime(&self) -> bool {
        Self::valid_state(self.v, self.w)
            && self.g_fast.is_finite()
            && self.g_fast >= 0.0
            && self.g_slow.is_finite()
            && self.g_slow >= 0.0
            && self.g_l.is_finite()
            && self.g_l >= 0.0
            && self.e_fast.is_finite()
            && self.e_slow.is_finite()
            && self.e_l.is_finite()
            && self.beta_w.is_finite()
            && self.gamma_w.is_finite()
            && self.gamma_w > 0.0
            && self.tau_w.is_finite()
            && self.tau_w > 0.0
            && self.phi.is_finite()
            && self.phi >= 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }

    fn derivatives(&self, v: f64, w: f64, i_ext: f64) -> Option<(f64, f64)> {
        if !Self::valid_state(v, w) {
            return None;
        }
        let m_inf = Self::sigmoid((v + 20.0) / 15.0);
        let w_inf = Self::sigmoid((v - self.beta_w) / self.gamma_w);
        let i_fast = self.g_fast * m_inf * (v - self.e_fast);
        let i_slow = self.g_slow * w * (v - self.e_slow);
        let i_l = self.g_l * (v - self.e_l);
        let dv = -i_fast - i_slow - i_l + i_ext;
        let dw = self.phi * (w_inf - w) / self.tau_w;
        if dv.is_finite() && dw.is_finite() {
            Some((dv, dw))
        } else {
            None
        }
    }

    fn rk4_step(&self, i_ext: f64) -> Option<(f64, f64)> {
        let dt = self.dt;
        let (k1_v, k1_w) = self.derivatives(self.v, self.w, i_ext)?;
        let (k2_v, k2_w) =
            self.derivatives(self.v + 0.5 * dt * k1_v, self.w + 0.5 * dt * k1_w, i_ext)?;
        let (k3_v, k3_w) =
            self.derivatives(self.v + 0.5 * dt * k2_v, self.w + 0.5 * dt * k2_w, i_ext)?;
        let (k4_v, k4_w) = self.derivatives(self.v + dt * k3_v, self.w + dt * k3_w, i_ext)?;
        let next_v = self.v + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
        let next_w = self.w + dt * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0;
        if Self::valid_state(next_v, next_w) {
            Some((next_v, next_w))
        } else {
            None
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !self.valid_runtime() {
            return 0;
        }
        let v_prev = self.v;
        let Some((next_v, next_w)) = self.rk4_step(i_ext) else {
            return 0;
        };
        self.v = next_v;
        self.w = next_w;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.w = 0.0;
    }
}

impl Default for PrescottNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_prescott(state: &PrescottNeuron) -> bool {
    state.valid_runtime()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prescott_new() {
        let state = PrescottNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_prescott(&state));
    }

    #[test]
    fn test_prescott_rk4_reference_point() {
        let mut state = PrescottNeuron::new();
        assert_eq!(state.step(50.0), 0);
        assert!((state.v - (-44.498914201492525)).abs() < 1e-12);
        assert!((state.w - 1.4035864179018786e-05).abs() < 1e-17);
    }

    #[test]
    fn test_prescott_invalid_input_preserves_state() {
        let mut state = PrescottNeuron::new();
        let before = (state.v, state.w);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.w), before);
    }

    #[test]
    fn test_prescott_invalid_candidate_preserves_state() {
        let mut state = PrescottNeuron::new();
        state.g_fast = f64::INFINITY;
        let before = (state.v, state.w);
        assert_eq!(state.step(50.0), 0);
        assert_eq!((state.v, state.w), before);
    }
}
