// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for morris_lecar

#[derive(Debug, Clone)]
pub struct MorrisLecarNeuron {
    pub v: f64,
    pub w: f64,
    pub c_m: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
    pub v4: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl MorrisLecarNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0_f64,
            w: 0.0_f64,
            c_m: 20.0_f64,
            g_ca: 4.0_f64,
            g_k: 8.0_f64,
            g_l: 2.0_f64,
            e_ca: 120.0_f64,
            e_k: -84.0_f64,
            e_l: -60.0_f64,
            v1: -1.2_f64,
            v2: 18.0_f64,
            v3: 12.0_f64,
            v4: 17.4_f64,
            phi: 1.0_f64 / 15.0_f64,
            dt: 0.1_f64,
            v_threshold: 0.0_f64,
        }
    }

    pub fn _m_inf(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - self.v1) / self.v2).tanh())
    }

    pub fn _w_inf(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - self.v3) / self.v4).tanh())
    }

    pub fn _lam(&self, v: f64) -> f64 {
        self.phi * ((v - self.v3) / (2.0 * self.v4)).cosh()
    }

    fn rhs(&self, v: f64, w: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && w.is_finite() && current.is_finite() && (0.0..=1.0).contains(&w)) {
            return None;
        }
        let m_inf = self._m_inf(v);
        let w_inf = self._w_inf(v);
        let lam = self._lam(v);
        if !(m_inf.is_finite() && w_inf.is_finite() && lam.is_finite()) {
            return None;
        }
        let i_ca = self.g_ca * m_inf * (v - self.e_ca);
        let i_k = self.g_k * w * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_ca - i_k - i_l + current) / self.c_m;
        let dw = lam * (w_inf - w);
        if dv.is_finite() && dw.is_finite() {
            Some((dv, dw))
        } else {
            None
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !validate_morris_lecar(self) || !current.is_finite() {
            return -1;
        }
        let v_prev = self.v;
        let Some((k1_v, k1_w)) = self.rhs(self.v, self.w, current) else {
            return -1;
        };
        let Some((k2_v, k2_w)) = self.rhs(
            self.v + 0.5 * self.dt * k1_v,
            self.w + 0.5 * self.dt * k1_w,
            current,
        ) else {
            return -1;
        };
        let Some((k3_v, k3_w)) = self.rhs(
            self.v + 0.5 * self.dt * k2_v,
            self.w + 0.5 * self.dt * k2_w,
            current,
        ) else {
            return -1;
        };
        let Some((k4_v, k4_w)) =
            self.rhs(self.v + self.dt * k3_v, self.w + self.dt * k3_w, current)
        else {
            return -1;
        };
        let mut next = self.clone();
        next.v += self.dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
        next.w += self.dt * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0;
        if !validate_morris_lecar(&next) {
            return -1;
        }
        *self = next;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -60.0_f64;
        self.w = 0.0_f64;
        self.c_m = 20.0_f64;
        self.g_ca = 4.0_f64;
        self.g_k = 8.0_f64;
        self.g_l = 2.0_f64;
        self.e_ca = 120.0_f64;
        self.e_k = -84.0_f64;
        self.e_l = -60.0_f64;
        self.v1 = -1.2_f64;
        self.v2 = 18.0_f64;
        self.v3 = 12.0_f64;
        self.v4 = 17.4_f64;
        self.phi = 1.0_f64 / 15.0_f64;
        self.dt = 0.1_f64;
        self.v_threshold = 0.0_f64;
    }
}

impl Default for MorrisLecarNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_morris_lecar(state: &MorrisLecarNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.c_m.is_finite()
        && state.g_ca.is_finite()
        && state.g_k.is_finite()
        && state.g_l.is_finite()
        && state.e_ca.is_finite()
        && state.e_k.is_finite()
        && state.e_l.is_finite()
        && state.v1.is_finite()
        && state.v2.is_finite()
        && state.v3.is_finite()
        && state.v4.is_finite()
        && state.phi.is_finite()
        && state.dt.is_finite()
        && state.v_threshold.is_finite()
        && state.c_m > 0.0
        && state.g_ca > 0.0
        && state.g_k > 0.0
        && state.g_l > 0.0
        && state.v2 > 0.0
        && state.v4 > 0.0
        && state.phi > 0.0
        && state.dt > 0.0
        && (0.0..=1.0).contains(&state.w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rhs_for_test(n: &MorrisLecarNeuron, v: f64, w: f64, current: f64) -> (f64, f64) {
        let m_inf = 0.5 * (1.0 + ((v - n.v1) / n.v2).tanh());
        let w_inf = 0.5 * (1.0 + ((v - n.v3) / n.v4).tanh());
        let lam = n.phi * ((v - n.v3) / (2.0 * n.v4)).cosh();
        let i_ca = n.g_ca * m_inf * (v - n.e_ca);
        let i_k = n.g_k * w * (v - n.e_k);
        let i_l = n.g_l * (v - n.e_l);
        ((-i_ca - i_k - i_l + current) / n.c_m, lam * (w_inf - w))
    }

    #[test]
    fn test_morris_lecar_new() {
        let state = MorrisLecarNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_morris_lecar(&state));
    }

    #[test]
    fn test_morris_lecar_step_matches_rk4_current_balance() {
        let mut state = MorrisLecarNeuron::new();
        let v0 = state.v;
        let w0 = state.w;
        let current = 50.0;
        let (k1_v, k1_w) = rhs_for_test(&state, v0, w0, current);
        let (k2_v, k2_w) = rhs_for_test(
            &state,
            v0 + 0.5 * state.dt * k1_v,
            w0 + 0.5 * state.dt * k1_w,
            current,
        );
        let (k3_v, k3_w) = rhs_for_test(
            &state,
            v0 + 0.5 * state.dt * k2_v,
            w0 + 0.5 * state.dt * k2_w,
            current,
        );
        let (k4_v, k4_w) =
            rhs_for_test(&state, v0 + state.dt * k3_v, w0 + state.dt * k3_w, current);
        let expected_v = v0 + state.dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
        let expected_w = w0 + state.dt * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0;

        let spike = state.step(current);

        assert!(spike == 0 || spike == 1);
        assert!((state.v - expected_v).abs() < 1e-12);
        assert!((state.w - expected_w).abs() < 1e-12);
    }

    #[test]
    fn test_morris_lecar_rejects_invalid_state() {
        let mut state = MorrisLecarNeuron::new();
        state.c_m = 0.0;
        let before = state.clone();
        assert_eq!(state.step(50.0), -1);
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn test_morris_lecar_rejects_invalid_current_without_mutation() {
        let mut state = MorrisLecarNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn test_morris_lecar_rejects_overflow_candidate_without_mutation() {
        let mut state = MorrisLecarNeuron::new();
        state.v = 1.0e6;
        state.w = 0.25;
        let before = state.clone();
        assert_eq!(state.step(0.0), -1);
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/morris_lecar.py (single-step RK4): silent at zero drive,
        // three action potentials at I=50 over 2000 steps, five at I=100. Morris-Lecar
        // gating is tanh/cosh, so the trace is not bit-exact across libms; the spike
        // count is the stable observable and is the parity contract — not a
        // "spike is 0 or 1" smoke check. The Go and Julia kernels reproduce the same counts.
        for (current, want) in [(0.0_f64, 0_usize), (50.0, 3), (100.0, 5)] {
            let mut state = MorrisLecarNeuron::new();
            let spikes = (0..2000).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
