// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for dcn_neuron

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct DCNNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64,
    pub s: f64,
    pub r: f64,
    pub ca: f64,
    pub g_na: f64,
    pub g_nap: f64,
    pub g_k: f64,
    pub g_t: f64,
    pub g_ahp: f64,
    pub g_h: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,
    pub kd_ahp: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
    pub _sub_steps: f64,
}

impl DCNNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0_f64,
            h: 0.6_f64,
            n: 0.32_f64,
            p: 0.01_f64,
            s: 0.8_f64,
            r: 0.1_f64,
            ca: 0.05_f64,
            g_na: 35.0_f64,
            g_nap: 0.5_f64,
            g_k: 9.0_f64,
            g_t: 0.1_f64,
            g_ahp: 2.0_f64,
            g_h: 0.02_f64,
            g_l: 0.2_f64,
            e_na: 55.0_f64,
            e_k: -90.0_f64,
            e_ca: 120.0_f64,
            e_h: -40.0_f64,
            e_l: -65.0_f64,
            c_m: 1.0_f64,
            phi: 5.0_f64,
            tau_ca: 150.0_f64,
            kd_ahp: 0.5_f64,
            dt: 0.5_f64,
            v_threshold: -20.0_f64,
            gain: 1.0_f64,
            _sub_steps: 20.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_dcn_neuron(self) || !i_ext.is_finite() {
            return 0;
        }
        let input = self.gain * i_ext;
        let sub_steps = self._sub_steps as usize;
        let sub_dt = self.dt / self._sub_steps;
        let mut fired = 0;
        let (mut v, mut h, mut n, mut p, mut s_gate, mut r, mut ca) =
            (self.v, self.h, self.n, self.p, self.s, self.r, self.ca);
        for _ in 0..sub_steps {
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);
            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();
            let p_inf = 1.0 / (1.0 + (-(v + 48.0) / 5.0).exp());
            let tau_p = 5.0 + 15.0 / (1.0 + ((v + 48.0) / 10.0).powi(2)).max(0.01);
            let mt_inf = 1.0 / (1.0 + (-(v + 52.0) / 5.0).exp());
            let s_inf = 1.0 / (1.0 + ((v + 60.0) / 6.5).exp());
            let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).exp());
            let r_inf = 1.0 / (1.0 + ((v + 80.0) / 10.0).exp());
            let tau_r = 100.0 + 200.0 / (1.0 + ((v + 70.0) / 10.0).exp());
            h = exact_hh_gate(h, alpha_h, beta_h, self.phi, sub_dt);
            n = exact_hh_gate(n, alpha_n, beta_n, self.phi, sub_dt);
            p = exact_relax(p, p_inf, tau_p, sub_dt);
            s_gate = exact_relax(s_gate, s_inf, tau_s, sub_dt);
            r = exact_relax(r, r_inf, tau_r, sub_dt);
            let i_t = self.g_t * mt_inf.powi(2) * s_gate * (v - self.e_ca);
            let ca_entry = if i_t < 0.0 { -i_t * 0.001 } else { 0.0 };
            ca = exact_relax(ca, ca_entry * self.tau_ca, self.tau_ca, sub_dt).max(0.0);
            let ahp_inf = ca.powi(2) / (ca.powi(2) + self.kd_ahp.powi(2));
            let g_na_eff = self.g_na * m_inf.powi(3) * h;
            let g_nap_eff = self.g_nap * p;
            let g_k_eff = self.g_k * n.powi(4);
            let g_t_eff = self.g_t * mt_inf.powi(2) * s_gate;
            let g_ahp_eff = self.g_ahp * ahp_inf;
            let g_h_eff = self.g_h * r;
            v = exact_voltage_step(
                v,
                input,
                self.c_m,
                sub_dt,
                &[
                    (g_na_eff, self.e_na),
                    (g_nap_eff, self.e_na),
                    (g_k_eff, self.e_k),
                    (g_t_eff, self.e_ca),
                    (g_ahp_eff, self.e_k),
                    (g_h_eff, self.e_h),
                    (self.g_l, self.e_l),
                ],
            );
            if v >= self.v_threshold {
                fired = 1;
                v = -60.0;
                s_gate *= 0.5;
                ca += 0.5;
            }
        }
        if ![v, h, n, p, s_gate, r, ca]
            .iter()
            .all(|value| value.is_finite())
        {
            return 0;
        }
        self.v = v.clamp(-100.0, 60.0);
        self.h = h.clamp(0.0, 1.0);
        self.n = n.clamp(0.0, 1.0);
        self.p = p.clamp(0.0, 1.0);
        self.s = s_gate.clamp(0.0, 1.0);
        self.r = r.clamp(0.0, 1.0);
        self.ca = ca.max(0.0);
        fired
    }

    pub fn reset(&mut self) {
        self.v = -60.0_f64;
        self.h = 0.6_f64;
        self.n = 0.32_f64;
        self.p = 0.01_f64;
        self.s = 0.8_f64;
        self.r = 0.1_f64;
        self.ca = 0.05_f64;
    }
}

pub fn validate_dcn_neuron(state: &DCNNeuron) -> bool {
    [
        state.v,
        state.h,
        state.n,
        state.p,
        state.s,
        state.r,
        state.ca,
        state.g_na,
        state.g_nap,
        state.g_k,
        state.g_t,
        state.g_ahp,
        state.g_h,
        state.g_l,
        state.e_na,
        state.e_k,
        state.e_ca,
        state.e_h,
        state.e_l,
        state.c_m,
        state.phi,
        state.tau_ca,
        state.kd_ahp,
        state.dt,
        state.v_threshold,
        state.gain,
        state._sub_steps,
    ]
    .iter()
    .all(|value| value.is_finite())
        && [state.h, state.n, state.p, state.s, state.r]
            .iter()
            .all(|gate| (0.0..=1.0).contains(gate))
        && state.ca >= 0.0
        && (-100.0..=60.0).contains(&state.v)
        && [
            state.g_na,
            state.g_nap,
            state.g_k,
            state.g_t,
            state.g_ahp,
            state.g_h,
            state.g_l,
        ]
        .iter()
        .all(|g| *g >= 0.0)
        && state.c_m > 0.0
        && state.phi > 0.0
        && state.tau_ca > 0.0
        && state.kd_ahp > 0.0
        && state.dt > 0.0
        && state.gain >= 0.0
        && state._sub_steps >= 1.0
}

fn safe_rate(a: f64, vhalf: f64, v: f64, k: f64, fallback: f64) -> f64 {
    let d = v + vhalf;
    if d.abs() < 1e-7 {
        fallback
    } else {
        a * d / (1.0 - (-d / k).exp())
    }
}

fn exact_relax(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
    target + (value - target) * (-dt / tau).exp()
}

fn exact_hh_gate(value: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> f64 {
    let rate = phi * (alpha + beta);
    let target = alpha / (alpha + beta);
    target + (value - target) * (-rate * dt).exp()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dcn_neuron_new() {
        let state = DCNNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_dcn_neuron(&state));
    }

    #[test]
    fn test_dcn_neuron_step() {
        let mut state = DCNNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
        assert!(state.v.is_finite());
        assert!(state.ca >= 0.0);
    }

    #[test]
    fn test_dcn_neuron_closed_form_gate_and_calcium_kinetics() {
        let mut state = DCNNeuron::new();
        state.g_na = 0.0;
        state.g_nap = 0.0;
        state.g_k = 0.0;
        state.g_t = 0.0;
        state.g_ahp = 0.0;
        state.g_h = 0.0;
        state.g_l = 0.0;
        state.gain = 0.0;
        let (v0, h0, n0, p0, s0, r0, ca0) = (
            state.v, state.h, state.n, state.p, state.s, state.r, state.ca,
        );
        let alpha_h = 0.07 * (-(v0 + 58.0) / 20.0).exp();
        let beta_h = 1.0 / (1.0 + (-(v0 + 28.0) / 10.0).exp());
        let alpha_n = safe_rate(0.01, 34.0, v0, 10.0, 0.1);
        let beta_n = 0.125 * (-(v0 + 44.0) / 80.0).exp();
        let p_inf = 1.0 / (1.0 + (-(v0 + 48.0) / 5.0).exp());
        let tau_p = 5.0 + 15.0 / (1.0 + ((v0 + 48.0) / 10.0).powi(2)).max(0.01);
        let s_inf = 1.0 / (1.0 + ((v0 + 60.0) / 6.5).exp());
        let tau_s = 20.0 + 50.0 / (1.0 + ((v0 + 65.0) / 10.0).exp());
        let r_inf = 1.0 / (1.0 + ((v0 + 80.0) / 10.0).exp());
        let tau_r = 100.0 + 200.0 / (1.0 + ((v0 + 70.0) / 10.0).exp());

        state.step(0.0);

        assert_close(state.v, v0);
        assert_close(
            state.h,
            exact_hh_gate(h0, alpha_h, beta_h, state.phi, state.dt),
        );
        assert_close(
            state.n,
            exact_hh_gate(n0, alpha_n, beta_n, state.phi, state.dt),
        );
        assert_close(state.p, exact_relax(p0, p_inf, tau_p, state.dt));
        assert_close(state.s, exact_relax(s0, s_inf, tau_s, state.dt));
        assert_close(state.r, exact_relax(r0, r_inf, tau_r, state.dt));
        assert_close(state.ca, exact_relax(ca0, 0.0, state.tau_ca, state.dt));
    }

    #[test]
    fn test_dcn_neuron_invalid_drive_preserves_state() {
        let mut state = DCNNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state.v, before.v);
        assert_eq!(state.ca, before.ca);
    }

    fn assert_close(observed: f64, expected: f64) {
        assert!(
            (observed - expected).abs() <= 1.0e-12,
            "observed {:.17e}, expected {:.17e}",
            observed,
            expected,
        );
    }
}
