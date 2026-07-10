// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for Izhikevich 2007

//! Standalone fail-closed mirror of the Izhikevich (2007) two-variable spiking
//! neuron in its physical (`Dynamical Systems in Neuroscience`, NeuroML 2
//! `izhikevich2007Cell`) parameterisation:
//!
//! ```text
//!   C v' = k (v - vr) (v - vt) - u + I
//!     u' = a (b (v - vr) - u)
//!   if v >= vpeak:  v <- c,  u <- u + d   (one spike)
//! ```
//!
//! `step` runs one candidate-first RK4 update — the production integrator the
//! Go/Julia/Mojo/Rust-engine backends implement — and rejects non-finite input
//! current or a non-finite candidate, preserving the previous state instead of
//! mutating it, so a poisoned trajectory never propagates. The arithmetic mirrors
//! `models/izhikevich2007.py` operation for operation (including the `dt / 6`
//! RK4 combination order), so the trace is bit-identical to the NumPy reference.

#[derive(Debug, Clone)]
pub struct Izhikevich2007Neuron {
    pub v: f64,
    pub u: f64,
    pub c_membrane: f64,
    pub k: f64,
    pub vr: f64,
    pub vt: f64,
    pub vpeak: f64,
    pub a: f64,
    pub b: f64,
    pub c_reset: f64,
    pub d: f64,
    pub dt: f64,
}

impl Izhikevich2007Neuron {
    pub fn new() -> Self {
        // Regular-spiking cortical parameters (Izhikevich 2007, Fig. 8.6). The
        // post-init state mirrors models/izhikevich2007.py reset_state():
        // v = vr, u = b (v - vr) = 0.
        Self {
            v: -60.0,
            u: 0.0,
            c_membrane: 100.0,
            k: 0.7,
            vr: -60.0,
            vt: -40.0,
            vpeak: 35.0,
            a: 0.03,
            b: -2.0,
            c_reset: -50.0,
            d: 100.0,
            dt: 0.1,
        }
    }

    fn rhs(&self, v: f64, u: f64, input_current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && u.is_finite() && input_current.is_finite()) {
            return None;
        }
        let dv = (self.k * (v - self.vr) * (v - self.vt) - u + input_current) / self.c_membrane;
        let du = self.a * (self.b * (v - self.vr) - u);
        if dv.is_finite() && du.is_finite() {
            Some((dv, du))
        } else {
            None
        }
    }

    fn rk4_candidate(&self, input_current: f64) -> Option<(f64, f64)> {
        let (v0, u0, dt) = (self.v, self.u, self.dt);
        let (k1v, k1u) = self.rhs(v0, u0, input_current)?;
        let (k2v, k2u) = self.rhs(v0 + 0.5 * dt * k1v, u0 + 0.5 * dt * k1u, input_current)?;
        let (k3v, k3u) = self.rhs(v0 + 0.5 * dt * k2v, u0 + 0.5 * dt * k2u, input_current)?;
        let (k4v, k4u) = self.rhs(v0 + dt * k3v, u0 + dt * k3u, input_current)?;
        // Match the Python `state + (dt / 6.0) * (k1 + 2 k2 + 2 k3 + k4)` order exactly:
        // the dt/6 factor is formed first, then multiplied into the weighted slope sum.
        let v = v0 + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        let u = u0 + (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        if v.is_finite() && u.is_finite() {
            Some((v, u))
        } else {
            None
        }
    }

    /// Advance one RK4 step. Returns `1` when the post-integration voltage reaches
    /// `vpeak` (emitting one spike and applying the reset `v <- c`, `u <- u + d`),
    /// `0` otherwise, and leaves the state untouched on a fail-closed rejection.
    pub fn step(&mut self, input_current: f64) -> i32 {
        if !validate_izhikevich2007(self) || !input_current.is_finite() {
            return 0;
        }
        let Some((v_new, u_new)) = self.rk4_candidate(input_current) else {
            return 0;
        };
        self.v = v_new;
        self.u = u_new;
        if self.v >= self.vpeak {
            self.v = self.c_reset;
            self.u += self.d;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        // Mirror models/izhikevich2007.py reset_state() at the default v0 (= vr):
        // restore only the state variables, never the parameters.
        self.v = self.vr;
        self.u = self.b * (self.v - self.vr);
    }
}

impl Default for Izhikevich2007Neuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_izhikevich2007(state: &Izhikevich2007Neuron) -> bool {
    state.v.is_finite()
        && state.u.is_finite()
        && state.c_membrane.is_finite()
        && state.k.is_finite()
        && state.vr.is_finite()
        && state.vt.is_finite()
        && state.vpeak.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.c_reset.is_finite()
        && state.d.is_finite()
        && state.dt.is_finite()
        && state.c_membrane > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rk4_reference(n: &Izhikevich2007Neuron, current: f64) -> (f64, f64) {
        let rhs = |v: f64, u: f64| {
            (
                (n.k * (v - n.vr) * (v - n.vt) - u + current) / n.c_membrane,
                n.a * (n.b * (v - n.vr) - u),
            )
        };
        let dt = n.dt;
        let (k1v, k1u) = rhs(n.v, n.u);
        let (k2v, k2u) = rhs(n.v + 0.5 * dt * k1v, n.u + 0.5 * dt * k1u);
        let (k3v, k3u) = rhs(n.v + 0.5 * dt * k2v, n.u + 0.5 * dt * k2u);
        let (k4v, k4u) = rhs(n.v + dt * k3v, n.u + dt * k3u);
        (
            n.v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
            n.u + (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u),
        )
    }

    #[test]
    fn test_izhikevich2007_new() {
        let state = Izhikevich2007Neuron::new();
        assert!(validate_izhikevich2007(&state));
        assert_eq!(state.v, -60.0);
        assert_eq!(state.u, 0.0);
    }

    #[test]
    fn test_izhikevich2007_matches_rk4_candidate() {
        // A sub-threshold drive: one step stays below vpeak, so the RK4 candidate is
        // committed unchanged and must match an independent RK4 re-derivation bit-for-bit.
        let mut state = Izhikevich2007Neuron::new();
        let expected = rk4_reference(&state, 50.0);
        assert_eq!(state.step(50.0), 0);
        assert!((state.v - expected.0).abs() < 1.0e-12);
        assert!((state.u - expected.1).abs() < 1.0e-12);
    }

    #[test]
    fn test_izhikevich2007_step() {
        let mut state = Izhikevich2007Neuron::new();
        let spike = state.step(100.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_izhikevich2007_spike_reset() {
        // Start near vpeak with a strong drive so the upstroke crosses threshold in one
        // step; the reset must set v to c_reset and add d to u.
        let mut state = Izhikevich2007Neuron::new();
        state.v = 30.0;
        state.u = 0.0;
        assert_eq!(state.step(1000.0), 1);
        assert_eq!(state.v, state.c_reset);
        assert!(state.u.is_finite());
        assert!(state.u >= state.d - 1.0);
    }

    #[test]
    fn test_izhikevich2007_invalid_current_preserves_state() {
        let mut state = Izhikevich2007Neuron::new();
        state.v = -55.0;
        state.u = 5.0;
        let before = (state.v, state.u);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.u), before);
    }

    #[test]
    fn test_izhikevich2007_overflow_candidate_preserves_state() {
        let mut state = Izhikevich2007Neuron::new();
        state.v = 1.0e200;
        let before = (state.v, state.u);
        // The quadratic drive term overflows, so the RK4 candidate is non-finite → fail-closed.
        assert_eq!(state.step(0.0), 0);
        assert_eq!((state.v, state.u), before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/izhikevich2007.py (RK4 integrator, default regular-spiking
        // parameters). The right-hand side is a quadratic in v plus a linear recovery variable —
        // all exact float arithmetic — so the trajectory is bit-for-bit across the Go, Julia and
        // Rust-engine backends, and the spike count (v reaching vpeak, then reset to c) is an exact
        // observable. Drive gates the regime around rheobase (~50-100 pA): silent at I=0.0, three
        // spikes at I=100, a fourteen-spike train at I=400, each over 2000 macro steps (dt=0.1 ms).
        // Verified python-vs-rust max|Δ|=0 via test_izhikevich2007_backends.py.
        for (current, want) in [(0.0_f64, 0_usize), (100.0, 3), (400.0, 14)] {
            let mut state = Izhikevich2007Neuron::new();
            let spikes = (0..2000).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
