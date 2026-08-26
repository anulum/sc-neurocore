// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for butera_respiratory

#[derive(Debug, Clone, PartialEq)]
pub struct ButeraRespiratoryNeuron {
    pub v: f64,
    pub n: f64,
    pub h_nap: f64,
    pub g_na: f64,
    pub g_nap: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub capacitance: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub g_tonic: f64,
    pub e_syn: f64,
    pub tau_h: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
struct Deriv {
    v: f64,
    n: f64,
    h_nap: f64,
}

impl ButeraRespiratoryNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.01,
            h_nap: 0.5,
            g_na: 28.0,
            g_nap: 2.8,
            g_k: 11.2,
            g_l: 2.8,
            capacitance: 21.0,
            e_na: 50.0,
            e_k: -85.0,
            e_l: -65.0,
            g_tonic: 0.0,
            e_syn: 0.0,
            tau_h: 10000.0,
            dt: 0.1,
            v_threshold: -20.0,
        }
    }

    fn valid_state(v: f64, n: f64, h_nap: f64) -> bool {
        [v, n, h_nap].iter().all(|x| x.is_finite())
            && (-200.0..=100.0).contains(&v)
            && (-0.05..=1.05).contains(&n)
            && (-0.05..=1.05).contains(&h_nap)
    }

    fn valid_static(&self) -> bool {
        [
            self.g_na,
            self.g_nap,
            self.g_k,
            self.g_l,
            self.capacitance,
            self.e_na,
            self.e_k,
            self.e_l,
            self.g_tonic,
            self.e_syn,
            self.tau_h,
            self.dt,
            self.v_threshold,
        ]
        .iter()
        .all(|x| x.is_finite())
            && self.g_na >= 0.0
            && self.g_nap >= 0.0
            && self.g_k >= 0.0
            && self.g_l >= 0.0
            && self.capacitance > 0.0
            && self.g_tonic >= 0.0
            && self.tau_h > 0.0
            && self.dt > 0.0
    }

    fn derivatives(&self, state: Deriv, current: f64) -> Option<Deriv> {
        let mut state = state;
        if !(state.v.is_finite()
            && state.n.is_finite()
            && state.h_nap.is_finite()
            && current.is_finite())
        {
            return None;
        }
        state.v = state.v.clamp(-200.0, 100.0);
        state.n = state.n.clamp(0.0, 1.0);
        state.h_nap = state.h_nap.clamp(0.0, 1.0);
        let m_na = 1.0 / (1.0 + (-(state.v + 34.0) / 5.0).exp());
        let m_nap = 1.0 / (1.0 + (-(state.v + 40.0) / 6.0).exp());
        let h_inf = 1.0 / (1.0 + ((state.v + 48.0) / 6.0).exp());
        let n_inf = 1.0 / (1.0 + (-(state.v + 29.0) / 4.0).exp());
        let tau_n = (10.0 / ((state.v + 29.0) / 8.0).cosh().max(1e-12)).max(0.01);
        let tau_h = (self.tau_h / ((state.v + 48.0) / 12.0).cosh().max(1e-12)).max(0.1);
        let i_na = self.g_na * m_na.powi(3) * (1.0 - state.n) * (state.v - self.e_na);
        let i_nap = self.g_nap * m_nap * state.h_nap * (state.v - self.e_na);
        let i_k = self.g_k * state.n.powi(4) * (state.v - self.e_k);
        let i_l = self.g_l * (state.v - self.e_l);
        let i_tonic = self.g_tonic * (state.v - self.e_syn);
        let deriv = Deriv {
            v: (-i_na - i_nap - i_k - i_l - i_tonic + current) / self.capacitance,
            n: (n_inf - state.n) / tau_n,
            h_nap: (h_inf - state.h_nap) / tau_h,
        };
        [deriv.v, deriv.n, deriv.h_nap]
            .iter()
            .all(|x| x.is_finite())
            .then_some(deriv)
    }

    fn rk4_candidate(&self, current: f64) -> Option<Deriv> {
        if !self.valid_static()
            || !current.is_finite()
            || !Self::valid_state(self.v, self.n, self.h_nap)
        {
            return None;
        }
        let state = Deriv {
            v: self.v,
            n: self.n,
            h_nap: self.h_nap,
        };
        let k1 = self.derivatives(state, current)?;
        let k2 = self.derivatives(
            Deriv {
                v: state.v + 0.5 * self.dt * k1.v,
                n: state.n + 0.5 * self.dt * k1.n,
                h_nap: state.h_nap + 0.5 * self.dt * k1.h_nap,
            },
            current,
        )?;
        let k3 = self.derivatives(
            Deriv {
                v: state.v + 0.5 * self.dt * k2.v,
                n: state.n + 0.5 * self.dt * k2.n,
                h_nap: state.h_nap + 0.5 * self.dt * k2.h_nap,
            },
            current,
        )?;
        let k4 = self.derivatives(
            Deriv {
                v: state.v + self.dt * k3.v,
                n: state.n + self.dt * k3.n,
                h_nap: state.h_nap + self.dt * k3.h_nap,
            },
            current,
        )?;
        let candidate = Deriv {
            v: state.v + self.dt * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v) / 6.0,
            n: state.n + self.dt * (k1.n + 2.0 * k2.n + 2.0 * k3.n + k4.n) / 6.0,
            h_nap: state.h_nap
                + self.dt * (k1.h_nap + 2.0 * k2.h_nap + 2.0 * k3.h_nap + k4.h_nap) / 6.0,
        };
        if candidate.v.is_finite() && candidate.n.is_finite() && candidate.h_nap.is_finite() {
            Some(Deriv {
                v: candidate.v.clamp(-200.0, 100.0),
                n: candidate.n.clamp(0.0, 1.0),
                h_nap: candidate.h_nap.clamp(0.0, 1.0),
            })
        } else {
            None
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        let v_prev = self.v;
        let Some(next) = self.rk4_candidate(i_ext) else {
            return 0;
        };
        self.v = next.v;
        self.n = next.n;
        self.h_nap = next.h_nap;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.01;
        self.h_nap = 0.5;
    }
}

pub fn validate_butera_respiratory(state: &ButeraRespiratoryNeuron) -> bool {
    state.valid_static() && ButeraRespiratoryNeuron::valid_state(state.v, state.n, state.h_nap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butera_respiratory_new() {
        let state = ButeraRespiratoryNeuron::new();
        assert!(validate_butera_respiratory(&state));
    }

    #[test]
    fn test_butera_respiratory_step() {
        let mut state = ButeraRespiratoryNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
        assert!(validate_butera_respiratory(&state));
    }

    #[test]
    fn invalid_current_preserves_state() {
        let mut state = ButeraRespiratoryNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state, before);
    }
}
