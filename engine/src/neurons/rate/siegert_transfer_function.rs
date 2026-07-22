// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Siegert transfer-function model

/// Siegert transfer function — analytical stationary firing rate of a LIF neuron.
#[derive(Clone, Debug)]
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
            tau_m: 20.0,
            tau_rp: 2.0,
            v_threshold: -50.0,
            v_reset: -70.0,
            v_rest: -65.0,
        }
    }
    pub fn step(&self, current: f64) -> f64 {
        let mu = self.v_rest + current;
        let sigma = current.abs().max(1e-6) * 0.1;
        let upper = (self.v_threshold - mu) / sigma;
        let lower = (self.v_reset - mu) / sigma;
        // Gauss-Legendre 20-point quadrature for ∫ exp(u²)(1+erf(u)) du
        let nodes = [
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
        let weights = [
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
        let half = (upper - lower) / 2.0;
        let mid = (upper + lower) / 2.0;
        let mut integral = 0.0;
        for (&node, &w) in nodes.iter().zip(weights.iter()) {
            let u = mid + half * node;
            let eu2 = (u * u).min(50.0).exp();
            let erf_u = Self::erf_approx(u);
            integral += w * eu2 * (1.0 + erf_u);
        }
        integral *= half;
        let t_isi = self.tau_rp + self.tau_m * std::f64::consts::PI.sqrt() * integral;
        1000.0 / t_isi.max(0.01)
    }
    fn erf_approx(x: f64) -> f64 {
        // Abramowitz-Stegun approximation
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let result = 1.0 - poly * (-x * x).exp();
        if x >= 0.0 {
            result
        } else {
            -result
        }
    }
}
impl Default for SiegertTransferFunction {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn siegert_rate() {
        let n = SiegertTransferFunction::new();
        let r = n.step(20.0);
        assert!(r > 0.0);
    }

    #[test]
    fn siegert_zero() {
        let n = SiegertTransferFunction::new();
        let r = n.step(0.0);
        assert!(r >= 0.0);
    }

    #[test]
    fn siegert_nan_no_panic() {
        SiegertTransferFunction::new().step(f64::NAN);
    }
}
