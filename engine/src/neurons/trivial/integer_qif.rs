// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Integer QIF Neuron

/// Wu et al. (2021) IQIF — piecewise-linear integer soma for digital hardware.
#[derive(Clone, Debug)]
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

impl IntegerQIFNeuron {
    #[allow(clippy::too_many_arguments)]
    pub fn with_parameters(
        v: i32,
        v_rest: i32,
        v_threshold: i32,
        v_reset: i32,
        a: i32,
        b: i32,
        v_max: i32,
        v_min: i32,
    ) -> Result<Self, &'static str> {
        let neuron = Self {
            v: i64::from(v),
            v_rest: i64::from(v_rest),
            v_threshold: i64::from(v_threshold),
            v_reset: i64::from(v_reset),
            a: i64::from(a),
            b: i64::from(b),
            v_max: i64::from(v_max),
            v_min: i64::from(v_min),
        };
        if neuron.valid() {
            Ok(neuron)
        } else {
            Err("invalid IQIF state or parameter ordering")
        }
    }

    pub fn valid(&self) -> bool {
        let i32_range = |value: i64| i64::from(i32::MIN) <= value && value <= i64::from(i32::MAX);
        [
            self.v,
            self.v_rest,
            self.v_threshold,
            self.v_reset,
            self.a,
            self.b,
            self.v_max,
            self.v_min,
        ]
        .into_iter()
        .all(i32_range)
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
        let denominator = self.a + self.b;
        if numerator >= 0 {
            numerator / denominator
        } else {
            -((-numerator) / denominator)
        }
    }

    pub fn try_step(&mut self, current: i32) -> Result<i32, &'static str> {
        if !self.valid() {
            return Err("invalid IQIF runtime state or parameter ordering");
        }
        let force = if self.v < self.branch_point() {
            self.a
                .checked_mul(self.v_rest - self.v)
                .ok_or("IQIF restoring force overflowed")?
        } else {
            self.b
                .checked_mul(self.v - self.v_threshold)
                .ok_or("IQIF restoring force overflowed")?
        };
        let candidate = self
            .v
            .checked_add(force >> 3)
            .and_then(|value| value.checked_add(i64::from(current)))
            .ok_or("IQIF candidate overflowed")?;
        if candidate > self.v_max {
            self.v = self.v_reset;
            Ok(1)
        } else {
            self.v = candidate.max(self.v_min);
            Ok(0)
        }
    }

    pub fn step(&mut self, current: i32) -> i32 {
        self.try_step(current)
            .expect("IQIF step requires a validated signed-int32 contract")
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iqif_fires() {
        let mut n = IntegerQIFNeuron::default();
        let total: i32 = (0..400).map(|_| n.step(10)).sum();
        assert_eq!(total, 26);
    }
    #[test]
    fn iqif_silent_without_input() {
        let mut n = IntegerQIFNeuron::default();
        let t: i32 = (0..200).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn iqif_reset_clears_state() {
        let mut n = IntegerQIFNeuron::default();
        for _ in 0..100 {
            n.step(10);
        }
        n.reset();
        assert_eq!(n.v, 128);
    }
    #[test]
    fn iqif_bounded() {
        let mut n = IntegerQIFNeuron::default();
        for _ in 0..1000 {
            n.step(10000);
        }
        // Integer — check no overflow panic occurred
    }
    #[test]
    fn iqif_negative_no_crash() {
        let mut n = IntegerQIFNeuron::default();
        for _ in 0..500 {
            n.step(-50);
        }
        assert_eq!(n.v, 0);
    }
    #[test]
    fn iqif_matches_source_tutorial_prefix_and_period() {
        let mut n = IntegerQIFNeuron::default();
        let mut trace = Vec::new();
        let mut spike_steps = Vec::new();
        for index in 0..400 {
            if n.step(10) == 1 {
                spike_steps.push(index);
            }
            trace.push(n.v);
        }
        assert_eq!(
            &trace[..15],
            &[138, 146, 153, 159, 165, 170, 176, 183, 190, 198, 207, 217, 229, 242, 128]
        );
        assert_eq!(spike_steps, (14..400).step_by(15).collect::<Vec<_>>());
        assert_eq!(trace.last(), Some(&198));
    }
}
