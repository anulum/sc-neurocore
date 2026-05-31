// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for stellate_cell

#[derive(Debug, Clone)]
pub struct StellateCell {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_kv3: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
    pub sub_steps: f64,
}

impl Default for StellateCell {
    fn default() -> Self {
        Self::new()
    }
}

impl StellateCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            p: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_kv3: 3.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 0.5,
            phi: 5.0,
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
            sub_steps: 50.0,
        }
    }

    fn safe_exp(value: f64) -> f64 {
        value.clamp(-60.0, 60.0).exp()
    }

    fn safe_rate(a: f64, vhalf: f64, v: f64, k: f64, fallback: f64) -> f64 {
        let d = v + vhalf;
        if d.abs() < 1e-7 {
            return fallback;
        }
        let z = -d / k;
        if z > 60.0 {
            0.0
        } else if z < -60.0 {
            a * d
        } else {
            a * d / (1.0 - z.exp())
        }
    }

    fn boltz(v: f64, vh: f64, k: f64) -> f64 {
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

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_stellate_cell(self) || !i_ext.is_finite() {
            return 0;
        }

        let sub_steps = self.sub_steps as usize;
        let inp = self.gain * i_ext;
        let sub_dt = self.dt / self.sub_steps;
        let mut fired = 0;
        let mut v = self.v;
        let mut h = self.h;
        let mut n = self.n;
        let mut p = self.p;

        for _ in 0..sub_steps {
            let alpha_m = Self::safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * Self::safe_exp(-(v + 60.0) / 18.0);
            let m_inf = alpha_m / (alpha_m + beta_m);
            let alpha_h = 0.07 * Self::safe_exp(-(v + 58.0) / 20.0);
            let beta_h = Self::boltz(v, -28.0, 10.0);
            let alpha_n = Self::safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * Self::safe_exp(-(v + 44.0) / 80.0);
            let p_inf = Self::boltz(v, -10.0, 10.0);
            let tau_p = 1.0 + 4.0 / (1.0 + Self::safe_exp((v + 20.0) / 15.0));

            h = Self::exact_hh_gate(h, alpha_h, beta_h, self.phi, sub_dt).clamp(0.0, 1.0);
            n = Self::exact_hh_gate(n, alpha_n, beta_n, self.phi, sub_dt).clamp(0.0, 1.0);
            p = Self::exact_relax(p, p_inf, tau_p, sub_dt).clamp(0.0, 1.0);

            let g_na_eff = self.g_na * m_inf.powi(3) * h;
            let g_k_eff = self.g_k * n.powi(4);
            let g_kv3_eff = self.g_kv3 * p.powi(2);
            v = Self::exact_voltage_step(
                v,
                inp,
                self.c_m,
                sub_dt,
                &[
                    (g_na_eff, self.e_na),
                    (g_k_eff, self.e_k),
                    (g_kv3_eff, self.e_k),
                    (self.g_l, self.e_l),
                ],
            )
            .clamp(-100.0, 60.0);
            if ![v, h, n, p].iter().all(|value| value.is_finite()) {
                return 0;
            }
            if v >= self.v_threshold {
                fired = 1;
                v = -65.0;
            }
        }

        self.v = v;
        self.h = h;
        self.n = n;
        self.p = p;
        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

pub fn validate_stellate_cell(state: &StellateCell) -> bool {
    [
        state.v,
        state.h,
        state.n,
        state.p,
        state.g_na,
        state.g_k,
        state.g_kv3,
        state.g_l,
        state.e_na,
        state.e_k,
        state.e_l,
        state.c_m,
        state.phi,
        state.dt,
        state.v_threshold,
        state.gain,
        state.sub_steps,
    ]
    .iter()
    .all(|value| value.is_finite())
        && [state.h, state.n, state.p]
            .iter()
            .all(|gate| (0.0..=1.0).contains(gate))
        && (-100.0..=60.0).contains(&state.v)
        && [state.g_na, state.g_k, state.g_kv3, state.g_l]
            .iter()
            .all(|conductance| *conductance >= 0.0)
        && state.c_m > 0.0
        && state.phi > 0.0
        && state.dt > 0.0
        && state.sub_steps > 0.0
        && state.sub_steps.fract() == 0.0
        && state.gain >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stellate_cell_new() {
        let state = StellateCell::new();
        assert!(state.v.is_finite());
        assert!(validate_stellate_cell(&state));
    }

    #[test]
    fn test_stellate_cell_step() {
        let mut state = StellateCell::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_stellate_cell_kv3_gate_activates_with_depolarisation() {
        let mut resting = StellateCell::new();
        let mut depolarised = StellateCell::new();
        for _ in 0..100 {
            resting.step(0.0);
            depolarised.step(8.0);
        }
        assert!(depolarised.p > resting.p);
    }

    #[test]
    fn test_stellate_cell_closed_form_gate_kinetics() {
        let mut state = StellateCell::new();
        state.g_na = 0.0;
        state.g_k = 0.0;
        state.g_kv3 = 0.0;
        state.g_l = 0.0;
        state.gain = 0.0;
        state.sub_steps = 1.0;
        let (v0, h0, n0, p0) = (state.v, state.h, state.n, state.p);
        let alpha_h = 0.07 * StellateCell::safe_exp(-(v0 + 58.0) / 20.0);
        let beta_h = StellateCell::boltz(v0, -28.0, 10.0);
        let alpha_n = StellateCell::safe_rate(0.01, 34.0, v0, 10.0, 0.1);
        let beta_n = 0.125 * StellateCell::safe_exp(-(v0 + 44.0) / 80.0);
        let p_inf = StellateCell::boltz(v0, -10.0, 10.0);
        let tau_p = 1.0 + 4.0 / (1.0 + StellateCell::safe_exp((v0 + 20.0) / 15.0));

        state.step(0.0);

        assert_close_stellate(state.v, v0);
        assert_close_stellate(
            state.h,
            StellateCell::exact_hh_gate(h0, alpha_h, beta_h, state.phi, state.dt),
        );
        assert_close_stellate(
            state.n,
            StellateCell::exact_hh_gate(n0, alpha_n, beta_n, state.phi, state.dt),
        );
        assert_close_stellate(
            state.p,
            StellateCell::exact_relax(p0, p_inf, tau_p, state.dt),
        );
    }

    #[test]
    fn test_stellate_cell_invalid_drive_preserves_state() {
        let mut state = StellateCell::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state.v, before.v);
        assert_eq!(state.h, before.h);
        assert_eq!(state.n, before.n);
        assert_eq!(state.p, before.p);
    }

    #[test]
    fn test_stellate_cell_corrupted_gate_preserves_state() {
        let mut state = StellateCell::new();
        state.h = -0.1;
        let before = state.clone();
        assert_eq!(state.step(8.0), 0);
        assert_eq!(state.v, before.v);
        assert_eq!(state.h, before.h);
        assert_eq!(state.n, before.n);
        assert_eq!(state.p, before.p);
    }

    fn assert_close_stellate(observed: f64, expected: f64) {
        assert!(
            (observed - expected).abs() <= 1.0e-12,
            "observed {:.17e}, expected {:.17e}",
            observed,
            expected,
        );
    }
}
