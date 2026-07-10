// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for rulkov_map

#[derive(Debug, Clone)]
pub struct RulkovMapNeuron {
    pub x: f64,
    pub y: f64,
    pub alpha: f64,
    pub sigma: f64,
    pub mu: f64,
    pub x_threshold: f64,
}

impl RulkovMapNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.0_f64,
            y: -3.0_f64,
            alpha: 4.0_f64,
            sigma: -1.6_f64,
            mu: 0.001_f64,
            x_threshold: 0.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_rulkov_map(self) {
            return Err("invalid Rulkov map runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Rulkov map current");
        }

        let x_prev = self.x;
        let branch_boundary = self.alpha + self.y + i_ext;
        if !branch_boundary.is_finite() {
            return Err("invalid Rulkov map branch boundary");
        }
        let x_new = if self.x <= 0.0 {
            let denominator = 1.0 - self.x;
            if denominator <= 0.0 || !denominator.is_finite() {
                return Err("invalid Rulkov map branch denominator");
            }
            self.alpha / denominator + self.y + i_ext
        } else if self.x < branch_boundary {
            branch_boundary
        } else {
            -1.0
        };
        let y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        if !x_new.is_finite() || !y_new.is_finite() {
            return Err("invalid Rulkov map candidate state");
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
        // Mirror models/rulkov_map.py `reset`: restore only the state variables,
        // never the parameters (alpha/sigma/mu are configuration, not state).
        self.x = -1.0_f64;
        self.y = -3.0_f64;
    }
}

impl Default for RulkovMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_rulkov_map(state: &RulkovMapNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.alpha.is_finite()
        && state.alpha > 0.0
        && state.sigma.is_finite()
        && state.mu.is_finite()
        && state.mu > 0.0
        && state.x_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rulkov_map_new() {
        let state = RulkovMapNeuron::new();
        assert!(validate_rulkov_map(&state));
    }

    #[test]
    fn test_rulkov_map_step() {
        let mut state = RulkovMapNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_rulkov_map_rejects_invalid_runtime_state() {
        let mut state = RulkovMapNeuron::new();
        state.y = f64::INFINITY;
        assert!(state.step(1.0).is_err());
    }

    // Independent re-derivation of one Rulkov map iteration, mirroring
    // models/rulkov_map.py step() exactly, to cross-check step() on all three
    // fast-map branches.
    fn map_reference(n: &RulkovMapNeuron, current: f64) -> (f64, f64) {
        let branch_boundary = n.alpha + n.y + current;
        let x_new = if n.x <= 0.0 {
            n.alpha / (1.0 - n.x) + n.y + current
        } else if n.x < branch_boundary {
            branch_boundary
        } else {
            -1.0
        };
        let y_new = n.y - n.mu * (n.x + 1.0) + n.mu * n.sigma;
        (x_new, y_new)
    }

    #[test]
    fn test_rulkov_map_matches_reference_all_branches() {
        // Branch 1 (x <= 0): piecewise-rational fast map.
        // Branch 2 (0 < x < alpha + y + I): plateau at the branch boundary.
        // Branch 3 (x >= alpha + y + I): hard reset of the fast variable to -1.
        for (x0, current) in [(-1.0_f64, 0.3), (0.5, 0.0), (2.0, 0.0)] {
            let mut state = RulkovMapNeuron {
                x: x0,
                ..RulkovMapNeuron::new()
            };
            let (xe, ye) = map_reference(&state, current);
            state.step(current).unwrap();
            assert!((state.x - xe).abs() < 1e-15, "x0={x0}");
            assert!((state.y - ye).abs() < 1e-15, "x0={x0}");
        }
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/rulkov_map.py (default parameters). The Rulkov 2001 map is a
        // discrete-time 2-D fast-slow iterated map (a piecewise-rational fast variable plus a
        // slow linear adaptation) — not an ODE, so there is no integrator; each `step` is one
        // exact map iteration. All arithmetic is a rational function plus linear updates, so the
        // orbit is bit-for-bit across languages and the spike count is an exact observable. The
        // slow adaptation carries the neuron out of the firing regime after an initial transient,
        // so the count saturates early (identical at 500 and 2000 iterations); drive gates it
        // cleanly: silent at I=0.0, four spikes at I=0.1, a 34-spike burst at I=0.5, each over
        // 2000 iterations. Verified python-vs-rust max|Δ|=0; the Go, Julia and Mojo backends
        // reproduce the same orbit via test_rulkov_map_backends.py.
        for (current, want) in [(0.0_f64, 0_usize), (0.1, 4), (0.5, 34)] {
            let mut state = RulkovMapNeuron::new();
            let spikes = (0..2000)
                .filter(|_| state.step(current).expect("finite step") == 1)
                .count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
