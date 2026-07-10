// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for wang_buzsaki

//! Wang-Buzsáki (1996) fast-spiking hippocampal interneuron.
//!
//! A three-state simplified Hodgkin-Huxley (Na + delayed-rectifier K only) with
//! instantaneous sodium activation `m = m_inf`. This kernel mirrors the Python golden
//! `sc_neurocore.neurons.models.wang_buzsaki.WangBuzsakiNeuron` operation-for-operation: a
//! 0.5 ms macro step of 50 inner `dt = 0.01` sub-steps advanced *sequentially*
//! (Gauss-Seidel) — the gating variables `h`/`n` are updated from the old voltage first,
//! then the membrane voltage `v` from the already-updated gates — with a rising-edge
//! `v >= v_threshold` crossing evaluated once on the macro boundary and no reset.
//!
//! Reference: Wang, X.-J. & Buzsáki, G. (1996), *Gamma oscillation by synaptic inhibition
//! in a hippocampal interneuronal network model*, J. Neurosci. 16:6402-6413,
//! DOI 10.1523/JNEUROSCI.16-20-06402.1996.

#[derive(Debug, Clone, PartialEq)]
pub struct WangBuzsakiNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl WangBuzsakiNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            g_na: 35.0,
            g_k: 9.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }

    /// `exp(x)` guarded against overflow to a non-finite value, matching the Python
    /// golden's `_safe_exp` fail-closed contract (the 700 bound keeps `exp` finite).
    fn safe_exp(x: f64) -> Option<f64> {
        if x.is_finite() && x <= 700.0 {
            Some(x.exp())
        } else {
            None
        }
    }

    /// Return `(m_inf, alpha_h, beta_h, alpha_n, beta_n)` — the gating rates at voltage `v`.
    ///
    /// Sodium activation is instantaneous: `m_inf = alpha_m / (alpha_m + beta_m)` with
    /// `alpha_m = 0.1*(v+35)/(1-exp(-(v+35)/10))` (its removable singularity at `v = -35`
    /// resolved to the limit `1.0`) and `beta_m = 4*exp(-(v+60)/18)`. The potassium rate
    /// `alpha_n = 0.01*(v+34)/(1-exp(-(v+34)/10))` carries the same removable singularity at
    /// `v = -34`. Transcribed verbatim from the Python golden's `_gating_rates`.
    fn gating_rates(v: f64) -> Option<(f64, f64, f64, f64, f64)> {
        let alpha_m = if (v + 35.0).abs() > 1e-6 {
            0.1 * (v + 35.0) / (1.0 - Self::safe_exp(-(v + 35.0) / 10.0)?)
        } else {
            1.0
        };
        let beta_m = 4.0 * Self::safe_exp(-(v + 60.0) / 18.0)?;
        let denom_m = alpha_m + beta_m;
        if denom_m == 0.0 || !denom_m.is_finite() {
            return None;
        }
        let m_inf = alpha_m / denom_m;
        let alpha_h = 0.07 * Self::safe_exp(-(v + 58.0) / 20.0)?;
        let beta_h = 1.0 / (1.0 + Self::safe_exp(-(v + 28.0) / 10.0)?);
        let alpha_n = if (v + 34.0).abs() > 1e-6 {
            0.01 * (v + 34.0) / (1.0 - Self::safe_exp(-(v + 34.0) / 10.0)?)
        } else {
            0.1
        };
        let beta_n = 0.125 * Self::safe_exp(-(v + 44.0) / 80.0)?;
        let rates = (m_inf, alpha_h, beta_h, alpha_n, beta_n);
        [m_inf, alpha_h, beta_h, alpha_n, beta_n]
            .iter()
            .all(|x| x.is_finite())
            .then_some(rates)
    }

    /// Advance one 0.5 ms macro step (50 sequential sub-steps) and report a spike.
    ///
    /// Returns `Ok(1)` on a rising-edge `v >= v_threshold` crossing over the macro step,
    /// `Ok(0)` otherwise, and `Err` if the input or the current state is invalid or the
    /// integration diverges — in which case the state is left unchanged (fail-closed: all
    /// mutation happens through locals and is only committed once the whole macro step is
    /// finite).
    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_wang_buzsaki(self) || !i_ext.is_finite() {
            return Err("invalid Wang-Buzsaki state or input");
        }
        let v_prev = self.v;
        let (mut v, mut h, mut n) = (self.v, self.h, self.n);
        let substeps = (0.5 / self.dt.max(0.001)) as usize;
        for _ in 0..substeps {
            // Gauss-Seidel order: gates from the old voltage, then voltage from the new gates.
            let Some((m_inf, alpha_h, beta_h, alpha_n, beta_n)) = Self::gating_rates(v) else {
                return Err("Wang-Buzsaki gating rate became non-finite");
            };
            let next_h = h + self.phi * (alpha_h * (1.0 - h) - beta_h * h) * self.dt;
            let next_n = n + self.phi * (alpha_n * (1.0 - n) - beta_n * n) * self.dt;
            let i_na = self.g_na * m_inf.powi(3) * next_h * (v - self.e_na);
            let i_k = self.g_k * next_n.powi(4) * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);
            let next_v = v + (-i_na - i_k - i_l + i_ext) / self.c_m * self.dt;
            if !next_v.is_finite() || !next_h.is_finite() || !next_n.is_finite() {
                return Err("Wang-Buzsaki state became non-finite");
            }
            v = next_v;
            h = next_h;
            n = next_n;
        }
        self.v = v;
        self.h = h;
        self.n = n;
        Ok((self.v >= self.v_threshold && v_prev < self.v_threshold) as i32)
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.8;
        self.n = 0.1;
    }
}

pub fn validate_wang_buzsaki(state: &WangBuzsakiNeuron) -> bool {
    state.v.is_finite()
        && state.h.is_finite()
        && state.n.is_finite()
        && state.g_na.is_finite()
        && state.g_na > 0.0
        && state.g_k.is_finite()
        && state.g_k > 0.0
        && state.g_l.is_finite()
        && state.g_l > 0.0
        && state.e_na.is_finite()
        && state.e_k.is_finite()
        && state.e_l.is_finite()
        && state.c_m.is_finite()
        && state.c_m > 0.0
        && state.phi.is_finite()
        && state.phi > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wang_buzsaki_new() {
        let state = WangBuzsakiNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_wang_buzsaki(&state));
    }

    #[test]
    fn test_wang_buzsaki_step() {
        let mut state = WangBuzsakiNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
        assert!(validate_wang_buzsaki(&state));
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // The Python golden (models/wang_buzsaki.py `WangBuzsakiNeuron`) fires three action
        // potentials at I = 10 over 20 macro steps — the same count the gauss_seidel schema
        // runner and the Q16.16 RTL reproduce three-way exactly. This Rust kernel must match it.
        let mut state = WangBuzsakiNeuron::new();
        let spikes: i32 = (0..20).map(|_| state.step(10.0).unwrap()).sum();
        assert_eq!(
            spikes, 3,
            "Wang-Buzsaki Rust kernel must reproduce the Python golden (3 AP @ I=10, 20 macro steps)"
        );
    }

    #[test]
    fn silent_at_zero_current() {
        // Zero drive settles without firing, as in the Python golden.
        let mut state = WangBuzsakiNeuron::new();
        let spikes: i32 = (0..20).map(|_| state.step(0.0).unwrap()).sum();
        assert_eq!(spikes, 0);
        assert!(validate_wang_buzsaki(&state));
    }

    #[test]
    fn test_wang_buzsaki_rejects_invalid_runtime_state() {
        let mut state = WangBuzsakiNeuron::new();
        state.h = f64::INFINITY;
        assert_eq!(state.step(10.0), Err("invalid Wang-Buzsaki state or input"));
    }

    #[test]
    fn invalid_current_preserves_state() {
        let mut state = WangBuzsakiNeuron::new();
        let before = state.clone();
        assert_eq!(
            state.step(f64::NAN),
            Err("invalid Wang-Buzsaki state or input")
        );
        assert_eq!(state, before);
    }

    #[test]
    fn reset_restores_gates() {
        let mut state = WangBuzsakiNeuron::new();
        for _ in 0..10 {
            let _ = state.step(10.0);
        }
        state.reset();
        assert_eq!(state.v, -65.0);
        assert_eq!(state.h, 0.8);
        assert_eq!(state.n, 0.1);
        assert!(validate_wang_buzsaki(&state));
    }
}
