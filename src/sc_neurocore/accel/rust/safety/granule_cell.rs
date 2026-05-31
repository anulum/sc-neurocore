// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for granule_cell

#[derive(Debug, Clone)]
pub struct GranuleCell {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub a: f64,
    pub b: f64,
    pub m_t: f64,
    pub s: f64,
    pub ca: f64,
    pub r: f64,
    pub c_m: f64,
    pub g_na: f64,
    pub g_kdr: f64,
    pub g_ka: f64,
    pub g_t: f64,
    pub g_kca: f64,
    pub g_h: f64,
    pub g_l: f64,
    pub g_tonic: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64,
    pub e_l: f64,
    pub e_gaba: f64,
    pub tau_ca: f64,
    pub kd_kca: f64,
    pub dt: f64,
    pub sub_steps: f64,
    pub gain: f64,
}

impl Default for GranuleCell {
    fn default() -> Self {
        Self::new()
    }
}

impl GranuleCell {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            m: 0.02,
            h: 0.85,
            n: 0.05,
            a: 0.1,
            b: 0.8,
            m_t: 0.01,
            s: 0.95,
            ca: 0.05,
            r: 0.1,
            c_m: 1.0,
            g_na: 17.0,
            g_kdr: 9.0,
            g_ka: 1.0,
            g_t: 0.5,
            g_kca: 3.5,
            g_h: 0.03,
            g_l: 0.1,
            g_tonic: 0.2,
            e_na: 87.4,
            e_k: -84.7,
            e_ca: 129.3,
            e_h: -40.0,
            e_l: -58.0,
            e_gaba: -75.0,
            tau_ca: 10.0,
            kd_kca: 0.2,
            dt: 0.5,
            sub_steps: 4.0,
            gain: 1.0,
        }
    }

    pub fn _boltz(&self, v: f64, vh: f64, k: f64) -> f64 {
        let z = -(v - vh) / k;
        if z > 60.0 {
            0.0
        } else if z < -60.0 {
            1.0
        } else {
            1.0 / (1.0 + z.exp())
        }
    }

    fn exact_relax(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn exact_voltage_step(
        v: f64,
        input_current: f64,
        c_m: f64,
        dt: f64,
        conductances: &[(f64, f64)],
    ) -> f64 {
        let g_total: f64 = conductances.iter().map(|(g, _)| *g).sum();
        if g_total <= 0.0 {
            return v + dt * input_current / c_m;
        }
        let reversal_drive: f64 = conductances.iter().map(|(g, e_rev)| g * e_rev).sum();
        let v_inf = (input_current + reversal_drive) / g_total;
        v_inf + (v - v_inf) * (-dt * g_total / c_m).exp()
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_granule_cell(self) || !i_ext.is_finite() {
            return 0;
        }

        let sub_steps = self.sub_steps as usize;
        let input = self.gain * i_ext;
        let dt_sub = self.dt / self.sub_steps;
        let v_prev = self.v;
        let mut v = self.v;
        let mut m = self.m;
        let mut h = self.h;
        let mut n = self.n;
        let mut a = self.a;
        let mut b = self.b;
        let mut m_t = self.m_t;
        let mut s_gate = self.s;
        let mut ca = self.ca;
        let mut r = self.r;

        for _ in 0..sub_steps {
            let m_inf = self._boltz(v, -30.0, 7.0);
            let tau_m = 0.1 + 0.3 / (1.0 + ((v + 30.0) / 10.0).powi(2)).max(0.01);
            m = Self::exact_relax(m, m_inf, tau_m, dt_sub).clamp(0.0, 1.0);

            let h_inf = self._boltz(v, -52.0, -6.0);
            let tau_h = 0.5 + 5.0 / (1.0 + ((v + 50.0) / 15.0).powi(2)).max(0.01);
            h = Self::exact_relax(h, h_inf, tau_h, dt_sub).clamp(0.0, 1.0);

            let n_inf = self._boltz(v, -35.0, 8.0);
            let tau_n = 1.0 + 5.0 / (1.0 + ((v + 35.0) / 15.0).powi(2)).max(0.01);
            n = Self::exact_relax(n, n_inf, tau_n, dt_sub).clamp(0.0, 1.0);

            let a_inf = self._boltz(v, -50.0, 20.0);
            a = Self::exact_relax(a, a_inf, 2.0, dt_sub).clamp(0.0, 1.0);

            let b_inf = self._boltz(v, -70.0, -6.0);
            b = Self::exact_relax(b, b_inf, 50.0, dt_sub).clamp(0.0, 1.0);

            let mt_inf = self._boltz(v, -52.0, 5.0);
            m_t = Self::exact_relax(m_t, mt_inf, 1.0, dt_sub).clamp(0.0, 1.0);

            let s_inf = self._boltz(v, -60.0, -6.5);
            let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).powi(2)).max(0.01);
            s_gate = Self::exact_relax(s_gate, s_inf, tau_s, dt_sub).clamp(0.0, 1.0);

            let r_inf = self._boltz(v, -80.0, -10.0);
            let tau_r = 50.0 + 200.0 / (1.0 + ((v + 80.0) / 20.0).powi(2)).max(0.01);
            r = Self::exact_relax(r, r_inf, tau_r, dt_sub).clamp(0.0, 1.0);

            let i_ca_t = self.g_t * m_t.powi(2) * s_gate * (v - self.e_ca);
            let ca_entry = if i_ca_t < 0.0 { -i_ca_t * 0.001 } else { 0.0 };
            ca = Self::exact_relax(ca, ca_entry * self.tau_ca, self.tau_ca, dt_sub).max(0.0);

            let kca_inf = ca * ca / (ca * ca + self.kd_kca * self.kd_kca);
            let g_na_eff = self.g_na * m.powi(3) * h;
            let g_kdr_eff = self.g_kdr * n.powi(4);
            let g_ka_eff = self.g_ka * a.powi(3) * b;
            let g_t_eff = self.g_t * m_t.powi(2) * s_gate;
            let g_kca_eff = self.g_kca * kca_inf;
            let g_h_eff = self.g_h * r;
            v = Self::exact_voltage_step(
                v,
                input,
                self.c_m,
                dt_sub,
                &[
                    (g_na_eff, self.e_na),
                    (g_kdr_eff, self.e_k),
                    (g_ka_eff, self.e_k),
                    (g_t_eff, self.e_ca),
                    (g_kca_eff, self.e_k),
                    (g_h_eff, self.e_h),
                    (self.g_l, self.e_l),
                    (self.g_tonic, self.e_gaba),
                ],
            )
            .clamp(-100.0, 60.0);

            if ![v, m, h, n, a, b, m_t, s_gate, ca, r]
                .iter()
                .all(|value| value.is_finite())
            {
                return 0;
            }
        }

        self.v = v;
        self.m = m;
        self.h = h;
        self.n = n;
        self.a = a;
        self.b = b;
        self.m_t = m_t;
        self.s = s_gate;
        self.ca = ca;
        self.r = r;

        if self.v >= 0.0 && v_prev < 0.0 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

pub fn validate_granule_cell(state: &GranuleCell) -> bool {
    [
        state.v,
        state.m,
        state.h,
        state.n,
        state.a,
        state.b,
        state.m_t,
        state.s,
        state.ca,
        state.r,
        state.c_m,
        state.g_na,
        state.g_kdr,
        state.g_ka,
        state.g_t,
        state.g_kca,
        state.g_h,
        state.g_l,
        state.g_tonic,
        state.e_na,
        state.e_k,
        state.e_ca,
        state.e_h,
        state.e_l,
        state.e_gaba,
        state.tau_ca,
        state.kd_kca,
        state.dt,
        state.sub_steps,
        state.gain,
    ]
    .iter()
    .all(|value| value.is_finite())
        && [
            state.m, state.h, state.n, state.a, state.b, state.m_t, state.s, state.r,
        ]
        .iter()
        .all(|gate| (0.0..=1.0).contains(gate))
        && (-100.0..=60.0).contains(&state.v)
        && state.ca >= 0.0
        && [
            state.g_na,
            state.g_kdr,
            state.g_ka,
            state.g_t,
            state.g_kca,
            state.g_h,
            state.g_l,
            state.g_tonic,
        ]
        .iter()
        .all(|conductance| *conductance >= 0.0)
        && state.c_m > 0.0
        && state.tau_ca > 0.0
        && state.kd_kca > 0.0
        && state.dt > 0.0
        && state.sub_steps > 0.0
        && state.sub_steps.fract() == 0.0
        && state.gain >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granule_cell_new() {
        let state = GranuleCell::new();
        assert!(state.v.is_finite());
        assert!(validate_granule_cell(&state));
    }

    #[test]
    fn test_granule_cell_step_preserves_bounds() {
        let mut state = GranuleCell::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
        assert!((-100.0..=60.0).contains(&state.v));
        assert!((0.0..=1.0).contains(&state.s));
        assert!(state.ca >= 0.0);
    }

    #[test]
    fn test_granule_cell_closed_form_gate_and_calcium_kinetics() {
        let mut state = GranuleCell::new();
        state.g_na = 0.0;
        state.g_kdr = 0.0;
        state.g_ka = 0.0;
        state.g_t = 0.0;
        state.g_kca = 0.0;
        state.g_h = 0.0;
        state.g_l = 0.0;
        state.g_tonic = 0.0;
        state.gain = 0.0;
        state.sub_steps = 1.0;
        let (v0, m0, h0, n0, a0, b0, mt0, s0, ca0, r0) = (
            state.v, state.m, state.h, state.n, state.a, state.b, state.m_t, state.s, state.ca,
            state.r,
        );
        let m_inf = state._boltz(v0, -30.0, 7.0);
        let tau_m = 0.1 + 0.3 / (1.0 + ((v0 + 30.0) / 10.0).powi(2)).max(0.01);
        let h_inf = state._boltz(v0, -52.0, -6.0);
        let tau_h = 0.5 + 5.0 / (1.0 + ((v0 + 50.0) / 15.0).powi(2)).max(0.01);
        let n_inf = state._boltz(v0, -35.0, 8.0);
        let tau_n = 1.0 + 5.0 / (1.0 + ((v0 + 35.0) / 15.0).powi(2)).max(0.01);
        let a_inf = state._boltz(v0, -50.0, 20.0);
        let b_inf = state._boltz(v0, -70.0, -6.0);
        let mt_inf = state._boltz(v0, -52.0, 5.0);
        let s_inf = state._boltz(v0, -60.0, -6.5);
        let tau_s = 20.0 + 50.0 / (1.0 + ((v0 + 65.0) / 10.0).powi(2)).max(0.01);
        let r_inf = state._boltz(v0, -80.0, -10.0);
        let tau_r = 50.0 + 200.0 / (1.0 + ((v0 + 80.0) / 20.0).powi(2)).max(0.01);

        state.step(0.0);

        assert_close_granule(state.v, v0);
        assert_close_granule(
            state.m,
            GranuleCell::exact_relax(m0, m_inf, tau_m, state.dt),
        );
        assert_close_granule(
            state.h,
            GranuleCell::exact_relax(h0, h_inf, tau_h, state.dt),
        );
        assert_close_granule(
            state.n,
            GranuleCell::exact_relax(n0, n_inf, tau_n, state.dt),
        );
        assert_close_granule(state.a, GranuleCell::exact_relax(a0, a_inf, 2.0, state.dt));
        assert_close_granule(state.b, GranuleCell::exact_relax(b0, b_inf, 50.0, state.dt));
        assert_close_granule(
            state.m_t,
            GranuleCell::exact_relax(mt0, mt_inf, 1.0, state.dt),
        );
        assert_close_granule(
            state.s,
            GranuleCell::exact_relax(s0, s_inf, tau_s, state.dt),
        );
        assert_close_granule(
            state.ca,
            GranuleCell::exact_relax(ca0, 0.0, state.tau_ca, state.dt),
        );
        assert_close_granule(
            state.r,
            GranuleCell::exact_relax(r0, r_inf, tau_r, state.dt),
        );
    }

    #[test]
    fn test_granule_cell_invalid_drive_preserves_state() {
        let mut state = GranuleCell::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state.v, before.v);
        assert_eq!(state.ca, before.ca);
        assert_eq!(state.s, before.s);
    }

    #[test]
    fn test_granule_cell_corrupted_gate_preserves_state() {
        let mut state = GranuleCell::new();
        state.m = -0.1;
        let before = state.clone();
        assert_eq!(state.step(10.0), 0);
        assert_eq!(state.v, before.v);
        assert_eq!(state.m, before.m);
        assert_eq!(state.ca, before.ca);
    }

    fn assert_close_granule(observed: f64, expected: f64) {
        assert!(
            (observed - expected).abs() <= 1.0e-12,
            "observed {:.17e}, expected {:.17e}",
            observed,
            expected,
        );
    }
}
