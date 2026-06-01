// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for wilson_cowan

#[derive(Debug, Clone)]
pub struct WilsonCowanUnit {
    pub e: f64,
    pub i: f64,
    pub w_ee: f64,
    pub w_ei: f64,
    pub w_ie: f64,
    pub w_ii: f64,
    pub tau_e: f64,
    pub tau_i: f64,
    pub a: f64,
    pub theta: f64,
    pub dt: f64,
}

impl WilsonCowanUnit {
    pub fn new() -> Self {
        Self {
            e: 0.1_f64,
            i: 0.05_f64,
            w_ee: 10.0_f64,
            w_ei: 6.0_f64,
            w_ie: 10.0_f64,
            w_ii: 1.0_f64,
            tau_e: 1.0_f64,
            tau_i: 2.0_f64,
            a: 1.2_f64,
            theta: 4.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn _sigmoid(&self, x: f64) -> f64 {
        logistic(self.a * (x - self.theta)) - logistic(-self.a * self.theta)
    }

    fn derivatives(&self, e: f64, i: f64, i_ext: f64) -> Result<(f64, f64), &'static str> {
        let se = self._sigmoid(self.w_ee * e - self.w_ei * i + i_ext);
        let si = self._sigmoid(self.w_ie * e - self.w_ii * i);
        let de = (-e + se) / self.tau_e;
        let di = (-i + si) / self.tau_i;
        if !de.is_finite() || !di.is_finite() {
            return Err("invalid Wilson-Cowan derivative");
        }
        Ok((de, di))
    }

    pub fn step(&mut self, i_ext: f64) -> Result<f64, &'static str> {
        if !validate_wilson_cowan(self) {
            return Err("invalid Wilson-Cowan runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Wilson-Cowan external input");
        }

        let (k1_e, k1_i) = self.derivatives(self.e, self.i, i_ext)?;
        let (k2_e, k2_i) = self.derivatives(
            self.e + 0.5 * self.dt * k1_e,
            self.i + 0.5 * self.dt * k1_i,
            i_ext,
        )?;
        let (k3_e, k3_i) = self.derivatives(
            self.e + 0.5 * self.dt * k2_e,
            self.i + 0.5 * self.dt * k2_i,
            i_ext,
        )?;
        let (k4_e, k4_i) =
            self.derivatives(self.e + self.dt * k3_e, self.i + self.dt * k3_i, i_ext)?;
        let next_e = self.e + self.dt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0;
        let next_i = self.i + self.dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0;
        if !finite_rate(next_e, self.a, self.theta) || !finite_rate(next_i, self.a, self.theta) {
            return Err("invalid Wilson-Cowan candidate state");
        }

        self.e = next_e;
        self.i = next_i;
        Ok(self.e)
    }

    pub fn reset(&mut self) {
        // self.e, self.i = 0.1, 0.05
        self.e = 0.1_f64;
        self.i = 0.05_f64;
        self.w_ee = 10.0_f64;
        self.w_ei = 6.0_f64;
        self.w_ie = 10.0_f64;
        self.w_ii = 1.0_f64;
        self.tau_e = 1.0_f64;
        self.tau_i = 2.0_f64;
        self.a = 1.2_f64;
        self.theta = 4.0_f64;
        self.dt = 0.1_f64;
    }
}

pub fn validate_wilson_cowan(state: &WilsonCowanUnit) -> bool {
    state.w_ee.is_finite()
        && state.w_ee >= 0.0
        && state.w_ei.is_finite()
        && state.w_ei >= 0.0
        && state.w_ie.is_finite()
        && state.w_ie >= 0.0
        && state.w_ii.is_finite()
        && state.w_ii >= 0.0
        && state.tau_e.is_finite()
        && state.tau_e > 0.0
        && state.tau_i.is_finite()
        && state.tau_i > 0.0
        && state.a.is_finite()
        && state.a > 0.0
        && state.theta.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && finite_rate(state.e, state.a, state.theta)
        && finite_rate(state.i, state.a, state.theta)
}

fn finite_rate(value: f64, a: f64, theta: f64) -> bool {
    let baseline = logistic(-a * theta);
    value.is_finite() && value >= -baseline && value <= 1.0 - baseline
}

fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let exp_z = z.exp();
        exp_z / (1.0 + exp_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wilson_cowan_new() {
        let state = WilsonCowanUnit::new();
        assert!(validate_wilson_cowan(&state));
    }

    #[test]
    fn test_wilson_cowan_step() {
        let mut state = WilsonCowanUnit::new();
        let rate = state.step(10.0).unwrap();
        assert!(rate.is_finite());
    }

    #[test]
    fn test_wilson_cowan_rejects_invalid_runtime_state() {
        let mut state = WilsonCowanUnit::new();
        state.e = 1.5;
        let before = state.clone();
        assert!(state.step(1.0).is_err());
        assert_eq!(state.e, before.e);
        assert_eq!(state.i, before.i);
    }

    #[test]
    fn test_wilson_cowan_step_uses_rk4_reference() {
        let mut state = WilsonCowanUnit::new();
        state.e = 0.24;
        state.i = 0.11;
        state.dt = 0.35;
        let rate = state.step(3.0).unwrap();
        let euler_e = 0.40111014473980233_f64;
        let euler_i = 0.10924537850891547_f64;
        assert!((rate - 0.42143718680097664_f64).abs() < 1e-15);
        assert!((state.i - 0.13798020053932203_f64).abs() < 1e-15);
        assert!((state.e - euler_e).abs() > 1e-2);
        assert!((state.i - euler_i).abs() > 1e-2);
    }
}
