// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Rust safety mirror for PersistentNaNeuron

#[derive(Debug, Clone)]
pub struct PersistentNaNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_nap: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
    pub sub_steps: usize,
}

impl PersistentNaNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            p: 0.0,
            g_na: 35.0,
            g_nap: 0.15,
            g_k: 9.0,
            g_l: 0.3,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
            sub_steps: 50,
        }
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !validate_persistent_na_neuron(self) {
            return Err("PersistentNa state and parameters must satisfy the public bounds");
        }

        let mut candidate = self.clone();
        let input = candidate.gain * current;
        let sub_dt = candidate.dt / candidate.sub_steps as f64;
        let mut fired = 0;
        for _ in 0..candidate.sub_steps {
            let v = candidate.v;
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);
            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();
            let p_inf = 1.0 / (1.0 + (-(v + 48.0) / 5.0).exp());
            let tau_p = 10.0 + 40.0 / (1.0 + ((v + 48.0) / 10.0).powi(2));

            candidate.h +=
                sub_dt * candidate.phi * (alpha_h * (1.0 - candidate.h) - beta_h * candidate.h);
            candidate.n +=
                sub_dt * candidate.phi * (alpha_n * (1.0 - candidate.n) - beta_n * candidate.n);
            candidate.p += sub_dt * (p_inf - candidate.p) / tau_p;
            let i_na = candidate.g_na * m_inf.powi(3) * candidate.h * (v - candidate.e_na);
            let i_nap = candidate.g_nap * candidate.p * (v - candidate.e_na);
            let i_k = candidate.g_k * candidate.n.powi(4) * (v - candidate.e_k);
            let i_l = candidate.g_l * (v - candidate.e_l);
            candidate.v += sub_dt * (-i_na - i_nap - i_k - i_l + input) / candidate.c_m;

            if ![candidate.v, candidate.h, candidate.n, candidate.p]
                .into_iter()
                .all(f64::is_finite)
            {
                return Err("PersistentNa candidate state became non-finite");
            }
            if candidate.v >= candidate.v_threshold {
                fired = 1;
                candidate.v = -65.0;
            }
        }

        candidate.v = candidate.v.clamp(-100.0, 60.0);
        candidate.h = candidate.h.clamp(0.0, 1.0);
        candidate.n = candidate.n.clamp(0.0, 1.0);
        candidate.p = candidate.p.clamp(0.0, 1.0);
        *self = candidate;
        Ok(fired)
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.6;
        self.n = 0.32;
        self.p = 0.0;
    }
}

fn safe_rate(a: f64, vhalf: f64, v: f64, k: f64, fallback: f64) -> f64 {
    let d = v + vhalf;
    if d.abs() < 1.0e-7 {
        fallback
    } else {
        a * d / (1.0 - (-d / k).exp())
    }
}

pub fn validate_persistent_na_neuron(state: &PersistentNaNeuron) -> bool {
    let finite = [
        state.v,
        state.h,
        state.n,
        state.p,
        state.g_na,
        state.g_nap,
        state.g_k,
        state.g_l,
        state.e_na,
        state.e_k,
        state.e_l,
        state.c_m,
        state.phi,
        state.dt,
        state.v_threshold,
        state.gain,
    ]
    .into_iter()
    .all(f64::is_finite);
    finite
        && (-100.0..=60.0).contains(&state.v)
        && [state.h, state.n, state.p]
            .into_iter()
            .all(|gate| (0.0..=1.0).contains(&gate))
        && (0.0..=200.0).contains(&state.g_na)
        && (0.0..=20.0).contains(&state.g_nap)
        && (0.0..=100.0).contains(&state.g_k)
        && (0.0..=5.0).contains(&state.g_l)
        && (30.0..=70.0).contains(&state.e_na)
        && (-100.0..=-70.0).contains(&state.e_k)
        && (-80.0..=-40.0).contains(&state.e_l)
        && (0.5..=2.0).contains(&state.c_m)
        && (0.5..=10.0).contains(&state.phi)
        && state.dt > 0.0
        && state.dt <= 1.0
        && (-20.0..=20.0).contains(&state.v_threshold)
        && (0.0..=10.0).contains(&state.gain)
        && (1..=10_000).contains(&state.sub_steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_step_matches_reference_anchor() {
        let mut state = PersistentNaNeuron::new();
        assert_eq!(state.step(5.0), Ok(0));
        assert!((state.v - -63.0).abs() < 3.0);
        assert!(state.h > 0.0 && state.h < 1.0);
        assert!(state.n > 0.0 && state.n < 1.0);
        assert!(state.p > 0.0 && state.p < 1.0);
    }

    #[test]
    fn invalid_drive_is_atomic() {
        let mut state = PersistentNaNeuron::new();
        let before = state.clone();
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.v, before.v);
        assert_eq!(state.p, before.p);
    }

    #[test]
    fn invalid_configuration_is_atomic() {
        let mut state = PersistentNaNeuron::new();
        state.c_m = 0.0;
        let before = state.clone();
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, before.v);
        assert_eq!(state.c_m, before.c_m);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = PersistentNaNeuron::new();
        state.g_nap = 0.3;
        state.v = -80.0;
        state.p = 0.7;
        state.reset();
        assert_eq!(state.v, -65.0);
        assert_eq!(state.p, 0.0);
        assert_eq!(state.g_nap, 0.3);
    }
}
