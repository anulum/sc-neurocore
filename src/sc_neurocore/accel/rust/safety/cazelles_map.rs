// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for cazelles_map

#[derive(Debug, Clone)]
pub struct CazellesMapNeuron {
    pub x: f64,
    pub y: f64,
    pub a: f64,
    pub epsilon: f64,
    pub sigma: f64,
    pub x_threshold: f64,
}

impl CazellesMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.1_f64,
            y: 0.0_f64,
            a: 3.8_f64,
            epsilon: 0.01_f64,
            sigma: 0.5_f64,
            x_threshold: 0.9_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_cazelles_map(self) || !i_ext.is_finite() {
            return 0;
        }
        // Cazelles logistic-driven map (models/cazelles_map.py step()):
        //   f     = a * x * (1 - x)
        //   x_new = clip(f - y + I, -2, 2)
        //   y_new = y + epsilon * (x - sigma)
        // The multiply/add stay separate IEEE operations (never fused) so the exact-arithmetic
        // backends reproduce the reference bit-for-bit.
        let f = self.a * self.x * (1.0 - self.x);
        let x_new = f - self.y + i_ext;
        let y_new = self.y + self.epsilon * (self.x - self.sigma);
        if !x_new.is_finite() || !y_new.is_finite() {
            return 0;
        }
        // clamp(-2, 2) equals Python min(2, max(-2, x_new)) bit-for-bit for finite inputs.
        self.x = x_new.clamp(-2.0, 2.0);
        self.y = y_new;
        i32::from(self.x >= self.x_threshold)
    }

    pub fn reset(&mut self) {
        // Mirror models/cazelles_map.py `reset`: restore only the state variables x and y,
        // never the parameters (a/epsilon/sigma are configuration, not state).
        self.x = 0.1_f64;
        self.y = 0.0_f64;
    }
}

impl Default for CazellesMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_cazelles_map(state: &CazellesMapNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.a.is_finite()
        && state.epsilon.is_finite()
        && state.sigma.is_finite()
        && state.x_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent re-derivation of one Cazelles map iteration, mirroring
    // models/cazelles_map.py step() exactly, to cross-check step().
    fn map_reference(n: &CazellesMapNeuron, current: f64) -> (f64, f64) {
        let f = n.a * n.x * (1.0 - n.x);
        let x_new = (f - n.y + current).clamp(-2.0, 2.0);
        let y_new = n.y + n.epsilon * (n.x - n.sigma);
        (x_new, y_new)
    }

    #[test]
    fn test_cazelles_map_new() {
        let state = CazellesMapNeuron::new();
        assert!(validate_cazelles_map(&state));
    }

    #[test]
    fn test_cazelles_map_step() {
        let mut state = CazellesMapNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_cazelles_map_matches_reference_interior_and_clamped() {
        // (0.3, 0.1) with I=0.2 stays inside [-2, 2]; (0.5, -2.0) with I=0 drives x_new past +2,
        // exercising the upper clamp.
        for (x0, y0, current) in [(0.3_f64, 0.1, 0.2), (0.5, -2.0, 0.0)] {
            let mut state = CazellesMapNeuron {
                x: x0,
                y: y0,
                ..CazellesMapNeuron::new()
            };
            let (xe, ye) = map_reference(&state, current);
            state.step(current);
            assert_eq!(state.x, xe, "x for x0={x0}");
            assert_eq!(state.y, ye, "y for x0={x0}");
            assert!((-2.0..=2.0).contains(&state.x), "clamped: x0={x0}");
        }
    }

    #[test]
    fn test_cazelles_map_invalid_current_preserves_state() {
        let mut state = CazellesMapNeuron::new();
        state.x = 0.42;
        state.y = 0.11;
        let before = (state.x, state.y);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.x, state.y), before);
    }

    #[test]
    fn test_cazelles_map_overflow_preserves_state() {
        let mut state = CazellesMapNeuron::new();
        // a * x * (1 - x) with x = -1e308 overflows the logistic term to -inf, so x_new is
        // non-finite → fail-closed, state untouched.
        state.x = -1.0e308;
        let before = (state.x, state.y);
        assert_eq!(state.step(0.0), 0);
        assert_eq!((state.x, state.y), before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/cazelles_map.py (default parameters). The Cazelles map is a
        // logistic-driven 2-D map in the chaotic regime (a = 3.8), folded through a hard clamp of
        // the fast variable to [-2, 2]. Every operation is exact IEEE arithmetic (multiply, add,
        // clamp), so the orbit is bit-for-bit across the exact-arithmetic backends (Go, Julia,
        // Rust) and the spike count — a level test x >= x_threshold on the clamped variable — is
        // an exact observable. The [-2, 2] clamp saturates the chaotic orbit, so unlike the pure
        // Medvedev map the count stays robust; the FMA-fusing Mojo backend, validated per-step
        // ULP-bounded, reproduces the same counts at the tested drives here. Drive gates the
        // count: 5 spikes at I=0.0, a 182-spike chaotic train at I=0.5, a 204-spike train at
        // I=1.0, each over 1000 iterations. Verified python-vs-rust max|Δ|=0.
        for (current, want) in [(0.0_f64, 5_usize), (0.5, 182), (1.0, 204)] {
            let mut state = CazellesMapNeuron::new();
            let spikes = (0..1000).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
