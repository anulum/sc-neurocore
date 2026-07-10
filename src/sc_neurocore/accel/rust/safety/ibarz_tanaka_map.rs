// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for ibarz_tanaka_map

#[derive(Debug, Clone)]
pub struct IbarzTanakaMapNeuron {
    pub x: f64,
    pub y: f64,
    pub alpha: f64,
    pub beta: f64,
    pub mu: f64,
    pub sigma: f64,
    pub x_threshold: f64,
    pub x_reset: f64,
}

impl IbarzTanakaMapNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.0_f64,
            y: -2.5_f64,
            alpha: 3.65_f64,
            beta: 0.25_f64,
            mu: 0.0005_f64,
            sigma: -1.6_f64,
            x_threshold: 3.0_f64,
            x_reset: -1.0_f64,
        }
    }

    /// Piecewise fast map `f` (Ibarz-Tanaka modified Rulkov map, eq. from
    /// models/ibarz_tanaka_map.py): a rational branch for the resting side and a
    /// linear branch for the depolarised side.
    pub fn f(&self, x: f64) -> f64 {
        if x <= 0.0 {
            // For x <= 0 the denominator 1 - x is >= 1, so the rational branch is
            // finite for any finite x (no division by a vanishing denominator).
            self.alpha / (1.0 - x)
        } else {
            self.alpha + self.beta * x
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_ibarz_tanaka_map(self) || !i_ext.is_finite() {
            return 0;
        }
        // Both updates use the OLD x (the Python computes x_new and y_new before
        // reassigning self.x), and the arithmetic stays as separate IEEE operations.
        let x_new = self.f(self.x) + self.y + i_ext;
        let y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        if !x_new.is_finite() || !y_new.is_finite() {
            return 0;
        }
        self.x = x_new;
        self.y = y_new;
        // Reset-on-spike: a threshold crossing hard-resets x to x_reset.
        if self.x >= self.x_threshold {
            self.x = self.x_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        // Mirror models/ibarz_tanaka_map.py `reset`: restore only the state
        // variables x and y, never the parameters.
        self.x = -1.0_f64;
        self.y = -2.5_f64;
    }
}

impl Default for IbarzTanakaMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_ibarz_tanaka_map(state: &IbarzTanakaMapNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.alpha.is_finite()
        && state.beta.is_finite()
        && state.mu.is_finite()
        && state.sigma.is_finite()
        && state.x_threshold.is_finite()
        && state.x_reset.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent re-derivation of one Ibarz-Tanaka map iteration, mirroring
    // models/ibarz_tanaka_map.py step() exactly (both updates use the old x).
    fn map_reference(n: &IbarzTanakaMapNeuron, current: f64) -> (f64, f64) {
        let fx = if n.x <= 0.0 {
            n.alpha / (1.0 - n.x)
        } else {
            n.alpha + n.beta * n.x
        };
        (
            fx + n.y + current,
            n.y - n.mu * (n.x + 1.0) + n.mu * n.sigma,
        )
    }

    #[test]
    fn test_ibarz_tanaka_map_new() {
        let state = IbarzTanakaMapNeuron::new();
        assert!(validate_ibarz_tanaka_map(&state));
    }

    #[test]
    fn test_ibarz_tanaka_map_step() {
        let mut state = IbarzTanakaMapNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_ibarz_tanaka_map_matches_reference_both_branches() {
        // x = -1 (<= 0): the rational fast branch. x = 0.5 (> 0): the linear branch. Both chosen
        // with a drive that keeps x_new below x_threshold, so no reset intervenes and the raw map
        // arithmetic can be checked exactly.
        for (x0, current) in [(-1.0_f64, 0.0), (0.5, 0.0)] {
            let mut state = IbarzTanakaMapNeuron {
                x: x0,
                ..IbarzTanakaMapNeuron::new()
            };
            let (xe, ye) = map_reference(&state, current);
            assert!(xe < state.x_threshold, "no spike for x0={x0}");
            state.step(current);
            assert_eq!(state.x, xe, "x for x0={x0}");
            assert_eq!(state.y, ye, "y for x0={x0}");
        }
    }

    #[test]
    fn test_ibarz_tanaka_map_spike_resets_to_x_reset() {
        // A strong drive pushes x_new past x_threshold, so the step must hard-reset x to x_reset.
        let mut state = IbarzTanakaMapNeuron {
            x: 0.5,
            y: 0.0,
            ..IbarzTanakaMapNeuron::new()
        };
        assert_eq!(state.step(3.0), 1);
        assert_eq!(state.x, state.x_reset);
    }

    #[test]
    fn test_ibarz_tanaka_map_invalid_current_preserves_state() {
        let mut state = IbarzTanakaMapNeuron::new();
        state.x = -0.4;
        state.y = -2.0;
        let before = (state.x, state.y);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.x, state.y), before);
    }

    #[test]
    fn test_ibarz_tanaka_map_overflow_preserves_state() {
        let mut state = IbarzTanakaMapNeuron::new();
        // The linear branch alpha + beta*x overflows to +inf → fail-closed, state untouched.
        state.x = 1.0e308;
        state.beta = 1.0e300;
        let before = (state.x, state.y);
        assert_eq!(state.step(0.0), 0);
        assert_eq!((state.x, state.y), before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/ibarz_tanaka_map.py (default parameters). The Ibarz-Tanaka map is a
        // modified Rulkov 2-D map (a rational/linear piecewise fast variable plus a slow linear
        // adaptation) with a reset-on-spike (x hard-resets to x_reset when it reaches x_threshold).
        // Every operation is exact IEEE arithmetic, so the orbit is bit-for-bit across the
        // exact-arithmetic backends (Go, Julia, Rust) and the spike count is an exact observable;
        // the per-spike reset re-synchronises the trajectory, so the FMA-fusing Mojo backend also
        // reproduces the same counts. Drive gates the regime around rheobase (~1.5): silent at
        // I=1.0, 69 spikes at I=1.5, a 235-spike burst train at I=2.0, each over 2000 iterations.
        // Verified python-vs-rust max|Δ|=0 (and python-vs-mojo counts equal) via
        // test_ibarz_tanaka_backends.py.
        for (current, want) in [(1.0_f64, 0_usize), (1.5, 69), (2.0, 235)] {
            let mut state = IbarzTanakaMapNeuron::new();
            let spikes = (0..2000).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
