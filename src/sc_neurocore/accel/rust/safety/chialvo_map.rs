// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for chialvo_map

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct ChialvoMapNeuron {
    pub x: f64,
    pub y: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub k: f64,
    pub x_threshold: f64,
}

impl ChialvoMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0_f64,
            y: 0.0_f64,
            a: 0.89_f64,
            b: 0.6_f64,
            c: 0.28_f64,
            k: 0.04_f64,
            x_threshold: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_chialvo_map(self) {
            return Err("invalid Chialvo map runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Chialvo map current");
        }

        let x_prev = self.x;
        let x_new = self.x.powi(2) * safe_exp(self.y - self.x) + self.k + i_ext;
        let y_new = self.a * self.y - self.b * self.x + self.c;
        if !x_new.is_finite() || !y_new.is_finite() {
            return Err("invalid Chialvo map candidate state");
        }
        self.x = x_new;
        self.y = y_new;
        Ok(if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        })
    }

    pub fn reset(&mut self) {
        // self.x, self.y = 0.0, 0.0
        self.x = 0.0_f64;
        self.y = 0.0_f64;
        self.a = 0.89_f64;
        self.b = 0.6_f64;
        self.c = 0.28_f64;
    }
}

pub fn validate_chialvo_map(state: &ChialvoMapNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.c.is_finite()
        && state.k.is_finite()
        && state.x_threshold.is_finite()
}

fn safe_exp(value: f64) -> f64 {
    value.clamp(-745.0, 709.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chialvo_map_new() {
        let state = ChialvoMapNeuron::new();
        assert!(validate_chialvo_map(&state));
    }

    #[test]
    fn test_chialvo_map_step() {
        let mut state = ChialvoMapNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_chialvo_map_rejects_invalid_runtime_state() {
        let mut state = ChialvoMapNeuron::new();
        state.y = f64::INFINITY;
        assert!(state.step(0.0).is_err());
    }
}
