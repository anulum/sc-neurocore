// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Rust safety contract for Wu et al. 2021 IQIF

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerQIFNeuron {
    pub v: i64,
    pub v_rest: i64,
    pub v_threshold: i64,
    pub v_reset: i64,
    pub a: i64,
    pub b: i64,
    pub v_max: i64,
    pub v_min: i64,
}

impl Default for IntegerQIFNeuron {
    fn default() -> Self {
        Self {
            v: 128,
            v_rest: 128,
            v_threshold: 200,
            v_reset: 128,
            a: 1,
            b: 1,
            v_max: 255,
            v_min: 0,
        }
    }
}

impl IntegerQIFNeuron {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn valid(&self) -> bool {
        let in_i32 = |value: i64| i64::from(i32::MIN) <= value && value <= i64::from(i32::MAX);
        [self.v, self.v_rest, self.v_threshold, self.v_reset, self.a, self.b, self.v_max, self.v_min]
            .into_iter()
            .all(in_i32)
            && self.a >= 0
            && self.b >= 0
            && self.a + self.b > 0
            && self.v_min < self.v_rest
            && self.v_rest < self.v_threshold
            && self.v_threshold < self.v_max
            && (self.v_min..=self.v_max).contains(&self.v_reset)
            && (self.v_min..=self.v_max).contains(&self.v)
    }

    pub fn branch_point(&self) -> i64 {
        let numerator = self.b * self.v_threshold + self.a * self.v_rest;
        if numerator >= 0 {
            numerator / (self.a + self.b)
        } else {
            -((-numerator) / (self.a + self.b))
        }
    }

    pub fn step(&mut self, current: i64) -> Result<i32, &'static str> {
        if !self.valid() || !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&current) {
            return Err("invalid IQIF signed-int32 contract");
        }
        let force = if self.v < self.branch_point() {
            self.a * (self.v_rest - self.v)
        } else {
            self.b * (self.v - self.v_threshold)
        };
        let candidate = self.v + (force >> 3) + current;
        if candidate > self.v_max {
            self.v = self.v_reset;
            Ok(1)
        } else {
            self.v = candidate.max(self.v_min);
            Ok(0)
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
}

pub fn validate_iqif(state: &IntegerQIFNeuron) -> bool {
    state.valid()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_pinned_source_tutorial() {
        let state = IntegerQIFNeuron::new();
        assert_eq!((state.v, state.v_rest, state.v_threshold), (128, 128, 200));
        assert_eq!((state.v_reset, state.a, state.b, state.v_max, state.v_min), (128, 1, 1, 255, 0));
        assert_eq!(state.branch_point(), 164);
        assert!(validate_iqif(&state));
    }

    #[test]
    fn source_trace_and_spikes_are_exact() {
        let mut state = IntegerQIFNeuron::new();
        let mut trace = Vec::new();
        let mut spikes = 0;
        for _ in 0..400 {
            spikes += state.step(10).unwrap();
            trace.push(state.v);
        }
        assert_eq!(&trace[..15], &[138, 146, 153, 159, 165, 170, 176, 183, 190, 198, 207, 217, 229, 242, 128]);
        assert_eq!(spikes, 26);
        assert_eq!(state.v, 198);
    }

    #[test]
    fn strict_upper_boundary_does_not_spike_on_equality() {
        let mut state = IntegerQIFNeuron::new();
        state.v = state.v_max;
        assert_eq!(state.step(-6), Ok(0));
        assert_eq!(state.v, state.v_max);
        assert_eq!(state.step(-5), Ok(1));
        assert_eq!(state.v, state.v_reset);
    }

    #[test]
    fn lower_candidate_is_clamped() {
        let mut state = IntegerQIFNeuron::new();
        state.v = 0;
        assert_eq!(state.step(-100), Ok(0));
        assert_eq!(state.v, 0);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = IntegerQIFNeuron { v: 140, v_rest: 100, v_threshold: 180, v_reset: 150, a: 2, b: 7, v_max: 250, v_min: 3 };
        state.reset();
        assert_eq!(state.v, 100);
        assert_eq!((state.v_reset, state.a, state.b, state.v_max), (150, 2, 7, 250));
    }

    #[test]
    fn invalid_coefficients_fail_before_mutation() {
        let mut state = IntegerQIFNeuron::new();
        state.a = -1;
        let before = state.v;
        assert!(state.step(10).is_err());
        assert_eq!(state.v, before);
    }

    #[test]
    fn zero_coefficient_profile_is_supported() {
        let mut state = IntegerQIFNeuron { a: 0, b: 3, ..IntegerQIFNeuron::new() };
        assert!(state.valid());
        assert_eq!(state.step(10), Ok(0));
    }

    #[test]
    fn out_of_int32_input_fails_closed() {
        let mut state = IntegerQIFNeuron::new();
        let before = state.clone();
        assert!(state.step(i64::from(i32::MAX) + 1).is_err());
        assert_eq!(state, before);
    }
}
