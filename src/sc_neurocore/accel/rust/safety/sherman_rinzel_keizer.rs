// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for Sherman-Rinzel-Keizer beta-cell burster

const TAU_N: f64 = 9.09;

#[derive(Debug, Clone)]
pub struct ShermanRinzelKeizerNeuron {
    pub v: f64,
    pub n: f64,
    pub s: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_s: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub tau_s: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

fn gate(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn sigmoid(arg: f64) -> f64 {
    1.0 / (1.0 + (-arg.clamp(-80.0, 80.0)).exp())
}

impl ShermanRinzelKeizerNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.1,
            s: 0.1,
            g_ca: 3.6,
            g_k: 10.0,
            g_s: 4.0,
            e_ca: 25.0,
            e_k: -75.0,
            tau_s: 5000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }

    fn valid(&self) -> bool {
        self.v.is_finite()
            && (-200.0..=200.0).contains(&self.v)
            && gate(self.n)
            && gate(self.s)
            && self.g_ca.is_finite()
            && self.g_ca > 0.0
            && self.g_k.is_finite()
            && self.g_k > 0.0
            && self.g_s.is_finite()
            && self.g_s >= 0.0
            && self.e_ca.is_finite()
            && self.e_k.is_finite()
            && self.tau_s.is_finite()
            && self.tau_s > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }

    fn derivatives(
        &self,
        v: f64,
        n_gate: f64,
        s_gate: f64,
        current: f64,
    ) -> Option<(f64, f64, f64)> {
        if !(v.is_finite() && n_gate.is_finite() && s_gate.is_finite() && current.is_finite()) {
            return None;
        }
        let m_inf = sigmoid((v + 20.0) / 12.0);
        let n_inf = sigmoid((v + 16.0) / 5.0);
        let s_inf = sigmoid((v + 35.0) / 10.0);
        let i_ca = self.g_ca * m_inf * (v - self.e_ca);
        let i_k = self.g_k * n_gate * (v - self.e_k);
        let i_s = self.g_s * s_gate * (v - self.e_k);
        let dv = -i_ca - i_k - i_s + current;
        let dn = (n_inf - n_gate) / TAU_N;
        let ds = (s_inf - s_gate) / self.tau_s;
        (dv.is_finite() && dn.is_finite() && ds.is_finite()).then_some((dv, dn, ds))
    }

    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64, f64)> {
        let half_dt = 0.5 * self.dt;
        let k1 = self.derivatives(self.v, self.n, self.s, current)?;
        let k2 = self.derivatives(
            self.v + half_dt * k1.0,
            self.n + half_dt * k1.1,
            self.s + half_dt * k1.2,
            current,
        )?;
        let k3 = self.derivatives(
            self.v + half_dt * k2.0,
            self.n + half_dt * k2.1,
            self.s + half_dt * k2.2,
            current,
        )?;
        let k4 = self.derivatives(
            self.v + self.dt * k3.0,
            self.n + self.dt * k3.1,
            self.s + self.dt * k3.2,
            current,
        )?;
        let next_v = self.v + self.dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        let next_n = self.n + self.dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
        let next_s = self.s + self.dt * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0;
        (next_v.is_finite() && (-200.0..=200.0).contains(&next_v) && gate(next_n) && gate(next_s))
            .then_some((next_v, next_n, next_s))
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid() || !current.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let Some((next_v, next_n, next_s)) = self.rk4_candidate(current) else {
            return 0;
        };
        self.v = next_v;
        self.n = next_n;
        self.s = next_s;
        i32::from(self.v >= self.v_threshold && v_prev < self.v_threshold)
    }

    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.1;
        self.s = 0.1;
    }
}

pub fn validate_sherman_rinzel_keizer(state: &ShermanRinzelKeizerNeuron) -> bool {
    state.valid()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srk_rk4_reference_point() {
        let mut state = ShermanRinzelKeizerNeuron::new();
        assert_eq!(state.step(5.0), 0);
        assert!((state.v - -54.24952703064663).abs() < 1e-12);
        assert!((state.n - 0.09468731121669713).abs() < 1e-12);
        assert!((state.s - 0.10000523900468992).abs() < 1e-12);
    }

    #[test]
    fn srk_invalid_state_preserves_values() {
        let mut state = ShermanRinzelKeizerNeuron::new();
        state.n = 1.2;
        assert_eq!(state.step(5.0), 0);
        assert_eq!(state.v, -50.0);
        assert_eq!(state.n, 1.2);
        assert_eq!(state.s, 0.1);
    }
}
