// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for spinnaker_lif

#[derive(Debug, Clone)]
pub struct SpiNNakerLIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub i_offset: f64,
    pub tau_refrac: f64,
    pub refrac_count: f64,
    pub dt: f64,
}

impl SpiNNakerLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0_f64,
            v_rest: -70.0_f64,
            v_reset: -70.0_f64,
            v_threshold: -50.0_f64,
            tau_m: 20.0_f64,
            i_offset: 0.0_f64,
            tau_refrac: 2.0_f64,
            refrac_count: 0.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_spinnaker_lif(self) || !i_ext.is_finite() {
            return -1;
        }
        if self.refrac_count > 0.0 {
            self.refrac_count = (self.refrac_count - self.dt).max(0.0);
            return 0;
        }

        let steady = self.v_rest + i_ext + self.i_offset;
        let next_v = steady + (self.v - steady) * (-self.dt / self.tau_m).exp();
        if !next_v.is_finite() {
            return -1;
        }
        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_count = self.tau_refrac;
            return 1;
        }
        self.v = next_v;
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refrac_count = 0.0_f64;
    }
}

pub fn validate_spinnaker_lif(state: &SpiNNakerLIFNeuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.v_threshold > state.v_reset
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.i_offset.is_finite()
        && state.tau_refrac.is_finite()
        && state.tau_refrac >= 0.0
        && state.refrac_count.is_finite()
        && state.refrac_count >= 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spinnaker_lif_new() {
        let state = SpiNNakerLIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_spinnaker_lif(&state));
    }

    #[test]
    fn test_spinnaker_lif_step() {
        let mut state = SpiNNakerLIFNeuron::new();
        let spike = state.step(30.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_spinnaker_lif_exact_flow() {
        let mut state = SpiNNakerLIFNeuron::new();
        let steady = state.v_rest + 10.0 + state.i_offset;
        let expected = steady + (state.v - steady) * (-state.dt / state.tau_m).exp();
        assert_eq!(state.step(10.0), 0);
        assert!((state.v - expected).abs() < 1.0e-12);
    }

    #[test]
    fn test_spinnaker_lif_rejects_invalid_current_without_mutation() {
        let mut state = SpiNNakerLIFNeuron::new();
        let original_v = state.v;
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.v, original_v);
    }
}
