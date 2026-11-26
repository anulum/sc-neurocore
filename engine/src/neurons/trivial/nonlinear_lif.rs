// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Nonlinear LIF Neuron

#![deny(missing_docs)]

/// Nonlinear LIF — quadratic membrane nonlinearity with adaptation. Touboul & Brette 2008.
#[derive(Clone, Debug)]
pub struct NonlinearLIFNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Adaptation current, in the units of the input current; it is subtracted from the
    /// membrane drive.
    pub w: f64,
    /// Resting membrane potential (mV).
    pub v_rest: f64,
    /// Critical spike-initiation potential (mV).
    pub v_crit: f64,
    /// Spike-detection threshold (mV).
    pub v_threshold: f64,
    /// Post-spike reset potential (mV).
    pub v_reset: f64,
    /// Quadratic curvature coefficient of the membrane nonlinearity (uA/(cm^2*mV^2)).
    pub a: f64,
    /// Subthreshold adaptation coupling strength (mS/cm^2).
    pub b: f64,
    /// Adaptation current time constant (ms).
    pub tau_w: f64,
    /// Membrane capacitance (uF/cm^2).
    pub c_m: f64,
    /// Integration step (ms).
    pub dt: f64,
}

impl NonlinearLIFNeuron {
    /// Create a neuron at rest (`v = v_rest`, `w = 0`) with the descriptor defaults and
    /// `dt` = 0.1 ms.
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_crit: -40.0,
            v_threshold: -20.0,
            v_reset: -65.0,
            a: 0.04,
            b: 0.5,
            tau_w: 100.0,
            c_m: 1.0,
            dt: 0.1,
        }
    }

    /// Advance one fourth-order Runge-Kutta step with the input `current` and return 1 on a
    /// spike, else 0.
    ///
    /// The derivatives are `dv = (a * (v - v_rest) * (v - v_crit) - w + current) / c_m` and
    /// `dw = (b * (v - v_rest) - w) / tau_w`. The candidate `(v, w)` is the classic RK4
    /// combination of four slopes, evaluated in the same order of operations as the Python
    /// model, the Go and Julia lanes and the safety crate, so the engine follows them bit for
    /// bit. The candidate becomes the state; when its `v` reaches `v_threshold`, `v` is set to
    /// `v_reset` and `w` is kept. Neither the input nor the parameters are validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let (next_v, next_w) = self.rk4_candidate(current);
        self.v = next_v;
        self.w = next_w;
        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    fn derivatives(&self, v: f64, w: f64, current: f64) -> (f64, f64) {
        let nonlinear = self.a * (v - self.v_rest) * (v - self.v_crit);
        let dv = (nonlinear - w + current) / self.c_m;
        let dw = (self.b * (v - self.v_rest) - w) / self.tau_w;
        (dv, dw)
    }

    fn rk4_candidate(&self, current: f64) -> (f64, f64) {
        let (k1_v, k1_w) = self.derivatives(self.v, self.w, current);
        let (k2_v, k2_w) = self.derivatives(
            self.v + 0.5 * self.dt * k1_v,
            self.w + 0.5 * self.dt * k1_w,
            current,
        );
        let (k3_v, k3_w) = self.derivatives(
            self.v + 0.5 * self.dt * k2_v,
            self.w + 0.5 * self.dt * k2_w,
            current,
        );
        let (k4_v, k4_w) =
            self.derivatives(self.v + self.dt * k3_v, self.w + self.dt * k3_w, current);
        let next_v = self.v + (self.dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        let next_w = self.w + (self.dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w);
        (next_v, next_w)
    }

    /// Return `v` to `v_rest` and clear `w`; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}

impl Default for NonlinearLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nlif_rk4_matches_the_python_model_bit_for_bit() {
        // Python's NonlinearLIFNeuron after 100 steps of 10.0 from its defaults: one spike,
        // v = -46.898989710403434 and w = 0.6505804659142335.
        let mut n = NonlinearLIFNeuron::new();
        let spikes: i32 = (0..100).map(|_| n.step(10.0)).sum();
        assert_eq!(spikes, 1);
        assert_eq!(n.v, -46.898989710403434);
        assert_eq!(n.w, 0.6505804659142335);
    }

    #[test]
    fn nlif_fires() {
        let mut n = NonlinearLIFNeuron::new();
        let total: i32 = (0..2000).map(|_| n.step(500.0)).sum();
        assert!(total > 0);
    }
    #[test]
    fn nlif_silent_without_input() {
        let mut n = NonlinearLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn nlif_reset_clears_state() {
        let mut n = NonlinearLIFNeuron::new();
        for _ in 0..100 {
            n.step(500.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn nlif_bounded() {
        let mut n = NonlinearLIFNeuron::new();
        for _ in 0..2000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn nlif_nan_no_panic() {
        NonlinearLIFNeuron::new().step(f64::NAN);
    }
    #[test]
    fn nlif_recovery_evolves() {
        let mut n = NonlinearLIFNeuron::new();
        for _ in 0..2000 {
            n.step(500.0);
        }
        assert!(
            n.w > 0.0,
            "recovery variable w should increase during spiking"
        );
    }
}
