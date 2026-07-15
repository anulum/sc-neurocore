// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Rust safety mirror for Jansen–Rit

/// Dependency-free equation-(6) state used by safety probes.
#[derive(Debug, Clone, PartialEq)]
pub struct JansenRitUnit {
    pub y0: f64,
    pub y3: f64,
    pub y1: f64,
    pub y4: f64,
    pub y2: f64,
    pub y5: f64,
    pub a_exc: f64,
    pub b_exc: f64,
    pub a_rate: f64,
    pub b_rate: f64,
    pub c: f64,
    pub e0: f64,
    pub v0: f64,
    pub r: f64,
    pub dt: f64,
}

impl JansenRitUnit {
    pub fn new() -> Self {
        Self {
            y0: 0.0,
            y3: 0.0,
            y1: 0.0,
            y4: 0.0,
            y2: 0.0,
            y5: 0.0,
            a_exc: 3.25,
            b_exc: 22.0,
            a_rate: 100.0,
            b_rate: 50.0,
            c: 135.0,
            e0: 2.5,
            v0: 6.0,
            r: 0.56,
            dt: 0.0001,
        }
    }

    fn sigmoid(&self, voltage: f64) -> Result<f64, String> {
        if !voltage.is_finite() {
            return Err("Jansen–Rit sigmoid input must be finite".into());
        }
        let exponent = self.r * (self.v0 - voltage);
        let response = if exponent >= 0.0 {
            let exp_neg = (-exponent).exp();
            2.0 * self.e0 * exp_neg / (1.0 + exp_neg)
        } else {
            2.0 * self.e0 / (1.0 + exponent.exp())
        };
        if response.is_finite() {
            Ok(response)
        } else {
            Err("Jansen–Rit sigmoid response must be finite".into())
        }
    }

    /// Advance atomically and return post-update `y1 - y2`.
    pub fn step(&mut self, p_ext: f64) -> Result<f64, String> {
        validate_jansen_rit(self)?;
        if !p_ext.is_finite() {
            return Err("Jansen–Rit external drive must be finite".into());
        }
        let c1 = self.c;
        let c2 = 0.8 * c1;
        let c3 = 0.25 * c1;
        let c4 = 0.25 * c1;
        let s_pyramidal = self.sigmoid(self.y1 - self.y2)?;
        let s_excitatory = self.sigmoid(c1 * self.y0)?;
        let s_inhibitory = self.sigmoid(c3 * self.y0)?;
        let dy0 = self.y3;
        let dy3 = self.a_exc * self.a_rate * s_pyramidal
            - 2.0 * self.a_rate * self.y3
            - self.a_rate.powi(2) * self.y0;
        let dy1 = self.y4;
        let dy4 = self.a_exc * self.a_rate * (p_ext + c2 * s_excitatory)
            - 2.0 * self.a_rate * self.y4
            - self.a_rate.powi(2) * self.y1;
        let dy2 = self.y5;
        let dy5 = self.b_exc * self.b_rate * c4 * s_inhibitory
            - 2.0 * self.b_rate * self.y5
            - self.b_rate.powi(2) * self.y2;
        let mut candidate = self.clone();
        candidate.y0 += dy0 * self.dt;
        candidate.y3 += dy3 * self.dt;
        candidate.y1 += dy1 * self.dt;
        candidate.y4 += dy4 * self.dt;
        candidate.y2 += dy2 * self.dt;
        candidate.y5 += dy5 * self.dt;
        validate_jansen_rit(&candidate)?;
        *self = candidate;
        Ok(self.y1 - self.y2)
    }

    pub fn reset(&mut self) {
        self.y0 = 0.0;
        self.y3 = 0.0;
        self.y1 = 0.0;
        self.y4 = 0.0;
        self.y2 = 0.0;
        self.y5 = 0.0;
    }
}

impl Default for JansenRitUnit {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_jansen_rit(state: &JansenRitUnit) -> Result<(), String> {
    let values = [
        state.y0,
        state.y3,
        state.y1,
        state.y4,
        state.y2,
        state.y5,
        state.a_exc,
        state.b_exc,
        state.a_rate,
        state.b_rate,
        state.c,
        state.e0,
        state.v0,
        state.r,
        state.dt,
    ];
    if !values.iter().all(|value| value.is_finite()) {
        return Err("Jansen–Rit state and parameters must be finite".into());
    }
    if state.a_exc <= 0.0
        || state.b_exc <= 0.0
        || state.a_rate <= 0.0
        || state.b_rate <= 0.0
        || state.e0 <= 0.0
        || state.r <= 0.0
        || state.dt <= 0.0
    {
        return Err("Jansen–Rit gains, rates, sigmoid scale, slope, and dt must be positive".into());
    }
    if state.c < 0.0 {
        return Err("Jansen–Rit connectivity must be non-negative".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_returns_continuous_eeg_and_preserves_atomicity() {
        let mut state = JansenRitUnit::new();
        assert!(state.step(220.0).unwrap().is_finite());
        let before = state.clone();
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state, before);
    }
}
