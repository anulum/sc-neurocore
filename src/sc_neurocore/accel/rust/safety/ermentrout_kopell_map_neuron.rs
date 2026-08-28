// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for ermentrout_kopell_map_neuron

#[derive(Debug, Clone)]
pub struct ErmentroutKopellMapNeuron {
    pub theta: f64,
    pub dt: f64,
    pub gain: f64,
    pub theta_threshold: f64,
}

impl ErmentroutKopellMapNeuron {
    pub fn new() -> Self {
        Self {
            theta: 0.0_f64,
            dt: 0.1_f64,
            gain: 1.0_f64,
            theta_threshold: std::f64::consts::PI,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_ermentrout_kopell_map_neuron(self) {
            return Err("invalid Ermentrout-Kopell runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Ermentrout-Kopell current");
        }

        let inp = self.gain * i_ext;
        if !inp.is_finite() {
            return Err("invalid Ermentrout-Kopell input drive");
        }
        let theta_prev = self.theta;
        let cos_theta = self.theta.cos();
        let d_theta = (1.0 - cos_theta) + (1.0 + cos_theta) * inp;
        let theta_next = self.theta + self.dt * d_theta;
        if !d_theta.is_finite() || !theta_next.is_finite() {
            return Err("invalid Ermentrout-Kopell candidate phase");
        }
        let fired = if theta_next >= self.theta_threshold && theta_prev < self.theta_threshold {
            1
        } else {
            0
        };
        self.theta = theta_next.rem_euclid(2.0 * std::f64::consts::PI);
        Ok(fired)
    }

    pub fn reset(&mut self) {
        // Mirror models/ermentrout_kopell_map_neuron.py `reset`: restore only the
        // phase state, never the parameters (dt/gain/theta_threshold are config).
        self.theta = 0.0_f64;
    }

    /// Checked complete-trace execution; rejected updates are atomic.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
        for _ in 0..n_steps {
            events += i64::from(self.step(current)?);
            trace.push(self.theta);
        }
        Ok((trace, events))
    }
}

impl Default for ErmentroutKopellMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_ermentrout_kopell_map_neuron(state: &ErmentroutKopellMapNeuron) -> bool {
    state.theta.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.gain.is_finite()
        && state.theta_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ermentrout_kopell_map_neuron_new() {
        let state = ErmentroutKopellMapNeuron::new();
        assert!(validate_ermentrout_kopell_map_neuron(&state));
    }

    #[test]
    fn test_ermentrout_kopell_map_neuron_step() {
        let mut state = ErmentroutKopellMapNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_ermentrout_kopell_map_neuron_rejects_invalid_runtime_state() {
        let mut state = ErmentroutKopellMapNeuron::new();
        state.theta = f64::INFINITY;
        assert!(state.step(1.0).is_err());
    }

    #[test]
    fn test_ermentrout_kopell_matches_reference_step() {
        // Independent re-derivation of one theta-neuron update (no threshold crossing), matched
        // bit-for-bit — the only transcendental is `cos`, and the test uses the same libm.
        let mut state = ErmentroutKopellMapNeuron::new();
        state.theta = 0.5;
        let inp = state.gain * 0.1;
        let d_theta = (1.0 - state.theta.cos()) + (1.0 + state.theta.cos()) * inp;
        let expected = (state.theta + state.dt * d_theta).rem_euclid(2.0 * std::f64::consts::PI);
        assert_eq!(state.step(0.1).unwrap(), 0);
        assert_eq!(state.theta, expected);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/ermentrout_kopell_map_neuron.py (default parameters). The
        // Ermentrout-Kopell theta neuron is the canonical Type-I phase model:
        // dtheta = (1 - cos theta) + (1 + cos theta) * gain * I, advanced by forward Euler and
        // wrapped modulo 2*pi, with a spike on an upward crossing of theta_threshold = pi. The only
        // transcendental is `cos`; on a shared libm the Rust trace is bit-for-bit with the NumPy
        // reference, and because the flow is a non-chaotic phase model the Go/Julia/Mojo libm
        // divergence is a non-amplifying sub-ULP band that never moves a pi-crossing — so the spike
        // count is the exact, portable observable across all backends. Drive gates the regime
        // cleanly: silent at I=-0.5 (below the SNIC bifurcation), 20 spikes at I=0.1, a 64-spike
        // train at I=1.0, each over 2000 macro steps. Verified python-vs-rust max|Δ|=0 (and
        // python-vs-mojo counts equal) via test_ermentrout_kopell_map_backends.py.
        for (current, want) in [(-0.5_f64, 0_usize), (0.1, 20), (1.0, 64)] {
            let mut state = ErmentroutKopellMapNeuron::new();
            let spikes = (0..2000)
                .filter(|_| state.step(current).expect("finite step") == 1)
                .count();
            assert_eq!(spikes, want, "I={current}");
        }
    }

    #[test]
    fn complete_trace_and_reset_preserve_the_public_contract() {
        let mut state = ErmentroutKopellMapNeuron {
            theta: 0.25,
            dt: 0.05,
            gain: 1.5,
            theta_threshold: 2.75,
        };
        let (trace, events) = state.simulate(512, 0.3).unwrap();
        assert_eq!(trace.len(), 512);
        assert!(events > 0);
        assert_eq!(trace.last().copied(), Some(state.theta));
        state.reset();
        assert_eq!(state.theta, 0.0);
        assert_eq!(
            (state.dt, state.gain, state.theta_threshold),
            (0.05, 1.5, 2.75)
        );
    }

    #[test]
    fn failed_batch_is_atomic_at_the_rejected_step() {
        let mut state = ErmentroutKopellMapNeuron::new();
        let before = state.theta;
        assert!(state.simulate(1, f64::NAN).is_err());
        assert_eq!(state.theta, before);
    }
}
