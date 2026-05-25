// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for siegert

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct SiegertTransferFunction {
    pub tau_m: f64,
    pub tau_rp: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub v_rest: f64,
}

impl SiegertTransferFunction {
    pub fn new() -> Self {
        Self {
            tau_m: 20.0_f64,
            tau_rp: 2.0_f64,
            v_threshold: -50.0_f64,
            v_reset: -70.0_f64,
            v_rest: -65.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<f64, &'static str> {
        if !validate_siegert(self) || !i_ext.is_finite() {
            return Err("siegert state/current must be finite and physically ordered");
        }
        let mu = self.v_rest + i_ext;
        if !mu.is_finite() {
            return Err("siegert mean voltage became non-finite");
        }
        let sigma = (i_ext.abs() * 0.1).max(1.0e-6);
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err("siegert diffusion scale became invalid");
        }
        let upper = (self.v_threshold - mu) / sigma;
        let lower = (self.v_reset - mu) / sigma;
        if !upper.is_finite() || !lower.is_finite() || upper <= lower {
            return Err("siegert first-passage bounds became invalid");
        }
        let half = (upper - lower) / 2.0;
        let mid = (upper + lower) / 2.0;
        if !half.is_finite() || !mid.is_finite() || half <= 0.0 {
            return Err("siegert quadrature interval became invalid");
        }
        let mut integral = 0.0;
        for (&node, &weight) in SIEGERT_NODES20.iter().zip(SIEGERT_WEIGHTS20.iter()) {
            let u = mid + half * node;
            let integrand = (u * u).min(50.0).exp() * (1.0 + siegert_erf_approx(u));
            if !integrand.is_finite() {
                return Err("siegert integrand became non-finite");
            }
            integral += weight * integrand;
        }
        integral *= half;
        if !integral.is_finite() || integral < 0.0 {
            return Err("siegert integral became invalid");
        }
        let t_isi = self.tau_rp + self.tau_m * std::f64::consts::PI.sqrt() * integral;
        if !t_isi.is_finite() || t_isi < self.tau_rp {
            return Err("siegert inter-spike interval became invalid");
        }
        let rate = 1000.0 / t_isi;
        let max_rate = 1000.0 / self.tau_rp;
        if !rate.is_finite() || rate < 0.0 || rate > max_rate {
            return Err("siegert rate became invalid");
        }
        Ok(rate)
    }

    pub fn reset(&mut self) {
        // pass
        self.tau_m = 20.0_f64;
        self.tau_rp = 2.0_f64;
        self.v_threshold = -50.0_f64;
        self.v_reset = -70.0_f64;
        self.v_rest = -65.0_f64;
    }
}

pub fn validate_siegert(state: &SiegertTransferFunction) -> bool {
    state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.tau_rp.is_finite()
        && state.tau_rp > 0.0
        && state.v_threshold.is_finite()
        && state.v_reset.is_finite()
        && state.v_rest.is_finite()
        && state.v_threshold > state.v_reset
}

const SIEGERT_NODES20: [f64; 20] = [
    -0.993128599185095,
    -0.963971927277914,
    -0.912234428251326,
    -0.839116971822219,
    -0.746331906460151,
    -0.636053680726515,
    -0.510867001950827,
    -0.373706088715420,
    -0.227785851141645,
    -0.076526521133497,
    0.076526521133497,
    0.227785851141645,
    0.373706088715420,
    0.510867001950827,
    0.636053680726515,
    0.746331906460151,
    0.839116971822219,
    0.912234428251326,
    0.963971927277914,
    0.993128599185095,
];

const SIEGERT_WEIGHTS20: [f64; 20] = [
    0.017614007139152,
    0.040601429800387,
    0.062672048334109,
    0.083276741576704,
    0.101930119817240,
    0.118194531961518,
    0.131688638449177,
    0.142096109318382,
    0.149172986472604,
    0.152753387130726,
    0.152753387130726,
    0.149172986472604,
    0.142096109318382,
    0.131688638449177,
    0.118194531961518,
    0.101930119817240,
    0.083276741576704,
    0.062672048334109,
    0.040601429800387,
    0.017614007139152,
];

fn siegert_erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-a * a).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_siegert_new() {
        let state = SiegertTransferFunction::new();
        assert!(validate_siegert(&state));
    }

    #[test]
    fn test_siegert_step() {
        let mut state = SiegertTransferFunction::new();
        let rate = state.step(20.0).unwrap();
        assert!(rate.is_finite());
        assert!(rate > 0.0);
    }

    #[test]
    fn test_siegert_rejects_invalid_runtime_state() {
        let mut state = SiegertTransferFunction::new();
        state.v_reset = state.v_threshold;
        assert!(state.step(20.0).is_err());
    }

    #[test]
    fn test_siegert_rate_is_refractory_bounded() {
        let mut state = SiegertTransferFunction::new();
        let rate = state.step(1.0e6).unwrap();
        assert!((0.0..=500.0).contains(&rate));
    }
}
