// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for galves_locherbach

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct GalvesLocherbachNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub decay: f64,
    pub threshold_rate: f64,
    pub steepness: f64,
    pub dt: f64,
}

impl GalvesLocherbachNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            v_rest: 0.0_f64,
            decay: 0.95_f64,
            threshold_rate: 0.5_f64,
            steepness: 5.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn _firing_prob(&self) -> f64 {
        let z = self.steepness * (self.v - self.threshold_rate);
        if z >= 0.0 {
            let tail = (-z).exp();
            1.0 / (1.0 + tail)
        } else {
            let tail = z.exp();
            tail / (1.0 + tail)
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_galves_locherbach(self) || !i_ext.is_finite() {
            return 0;
        }
        self.v = self.decay * self.v + i_ext;
        let p = self._firing_prob() * self.dt;
        if p >= 1.0 {
            self.v = self.v_rest;
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        // self.v = self.v_rest
        self.v = 0.0_f64;
        self.v_rest = 0.0_f64;
        self.decay = 0.95_f64;
        self.threshold_rate = 0.5_f64;
        self.steepness = 5.0_f64;
    }
}

pub fn validate_galves_locherbach(state: &GalvesLocherbachNeuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.threshold_rate.is_finite()
        && state.decay.is_finite()
        && (0.0..=1.0).contains(&state.decay)
        && state.steepness.is_finite()
        && state.steepness > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.dt <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galves_locherbach_new() {
        let state = GalvesLocherbachNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_galves_locherbach(&state));
    }

    #[test]
    fn test_galves_locherbach_step() {
        let mut state = GalvesLocherbachNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
