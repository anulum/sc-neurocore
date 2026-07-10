// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for medvedev_map

#[derive(Debug, Clone)]
pub struct MedvedevMapNeuron {
    pub x: f64,
    pub alpha: f64,
    pub beta: f64,
    pub x_threshold: f64,
}

impl MedvedevMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0_f64,
            alpha: 3.5_f64,
            beta: 0.5_f64,
            x_threshold: 0.9_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_medvedev_map(self) || !i_ext.is_finite() {
            return 0;
        }
        let x_prev = self.x;
        // Medvedev piecewise-linear expanding circle map (models/medvedev_map.py step()):
        //   x <- (alpha * x + I) mod 1        if x < beta
        //   x <- (alpha * (1 - x) + I) mod 1  otherwise
        // The multiply and add are kept as separate IEEE operations (never fused into a
        // multiply-add) so the fast branch matches the Python `alpha * x + current` bit-for-bit.
        let mapped = if self.x < self.beta {
            self.alpha * self.x + i_ext
        } else {
            self.alpha * (1.0 - self.x) + i_ext
        };
        if !mapped.is_finite() {
            return 0;
        }
        // Euclidean remainder: rem_euclid(1.0) is bit-identical to Python `x % 1.0` for the unit
        // divisor (both fold into [0, 1)), so the chaotic orbit reproduces the reference exactly.
        let x_new = mapped.rem_euclid(1.0);
        if !x_new.is_finite() {
            return 0;
        }
        self.x = x_new;
        i32::from(self.x >= self.x_threshold && x_prev < self.x_threshold)
    }

    pub fn reset(&mut self) {
        // Mirror models/medvedev_map.py `reset`: restore only the state variable x,
        // never the parameters (alpha/beta/x_threshold are configuration, not state).
        self.x = 0.0_f64;
    }
}

impl Default for MedvedevMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_medvedev_map(state: &MedvedevMapNeuron) -> bool {
    state.x.is_finite()
        && state.alpha.is_finite()
        && state.beta.is_finite()
        && state.x_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent re-derivation of one Medvedev map iteration, mirroring
    // models/medvedev_map.py step() exactly, to cross-check step() on both branches.
    fn map_reference(n: &MedvedevMapNeuron, current: f64) -> f64 {
        let mapped = if n.x < n.beta {
            n.alpha * n.x + current
        } else {
            n.alpha * (1.0 - n.x) + current
        };
        mapped.rem_euclid(1.0)
    }

    #[test]
    fn test_medvedev_map_new() {
        let state = MedvedevMapNeuron::new();
        assert!(validate_medvedev_map(&state));
    }

    #[test]
    fn test_medvedev_map_step() {
        let mut state = MedvedevMapNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_medvedev_map_matches_reference_both_branches() {
        // Branch x < beta: the linear-expansion arm.  Branch x >= beta: the reflected arm.
        // Both fold through the unit Euclidean remainder into [0, 1).
        for (x0, current) in [(0.3_f64, 0.2), (0.7, 0.1)] {
            let mut state = MedvedevMapNeuron {
                x: x0,
                ..MedvedevMapNeuron::new()
            };
            let expected = map_reference(&state, current);
            state.step(current);
            assert_eq!(state.x, expected, "x0={x0}");
            assert!((0.0..1.0).contains(&state.x), "folded into [0,1): x0={x0}");
        }
    }

    #[test]
    fn test_medvedev_map_invalid_current_preserves_state() {
        let mut state = MedvedevMapNeuron::new();
        state.x = 0.42;
        let before = state.x;
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state.x, before);
    }

    #[test]
    fn test_medvedev_map_overflow_preserves_state() {
        let mut state = MedvedevMapNeuron::new();
        state.x = 1.0e308;
        let before = state.x;
        // 3.5 * (1 - 1e308) overflows to -inf → fail-closed, state untouched.
        assert_eq!(state.step(0.0), 0);
        assert_eq!(state.x, before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/medvedev_map.py (default parameters). The Medvedev map is a
        // discrete-time piecewise-linear EXPANDING circle map (alpha = 3.5 > 1), so the orbit is
        // chaotic — yet fully deterministic and, because every operation is exact IEEE arithmetic
        // (multiply, add, Euclidean fold), bit-for-bit across the exact-arithmetic backends. The
        // spike count is therefore an exact observable HERE for the Rust safety kernel (and the
        // Go/Julia/Rust-engine lanes). NOTE: the FMA-fusing Mojo backend contracts alpha*x+I into
        // one rounding; on an expanding map that single-ULP difference is amplified, so Mojo does
        // NOT reproduce the exact spike count over long horizons (it is validated on a per-step
        // ULP bound and structural invariants instead — by design, see
        // test_medvedev_map_backends.py). Drive gates the regime: silent at I=0.0 (fixed point at
        // 0), a 92-spike chaotic train at I=0.2, a 112-spike train at I=0.5, each over 1000
        // iterations. Verified python-vs-rust max|Δ|=0.
        for (current, want) in [(0.0_f64, 0_usize), (0.2, 92), (0.5, 112)] {
            let mut state = MedvedevMapNeuron::new();
            let spikes = (0..1000).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
