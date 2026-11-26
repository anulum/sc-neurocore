// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — RK4 neuron integrator ports

//! Explicit RK4 ports for the priority neuron integrator paths.

#![deny(missing_docs)]

pub(crate) mod bindings;

/// The PyO3 simulator binding, kept at the path where 4.2.0 published it before it moved into
/// [`bindings`]; see [`bindings::py_rk4_neuron_simulate`].
pub use bindings::py_rk4_neuron_simulate;

const IZH_SPIKE_THRESHOLD: f64 = 30.0;

/// Izhikevich (2003) neuron, integrated with one classical RK4 step of `dt` per call.
///
/// `v' = 0.04 v^2 + 5 v + 140 - u + I` and `u' = a (b v - u)`, with the input held over the
/// step; `v >= 30` after the step sets `v` to `c` and adds `d` to `u`. Nothing is validated.
#[derive(Clone, Debug)]
pub struct IzhikevichRk4 {
    /// Membrane potential (mV).
    pub v: f64,
    /// Recovery variable.
    pub u: f64,
    /// Time scale of the recovery variable.
    pub a: f64,
    /// Sensitivity of the recovery variable to `v`.
    pub b: f64,
    /// Reset value of `v` after a spike (mV).
    pub c: f64,
    /// Increment of `u` at each spike.
    pub d: f64,
    /// RK4 step (ms).
    pub dt: f64,
}

impl IzhikevichRk4 {
    /// Create a regular-spiking neuron (a = 0.02, b = 0.2, c = -65, d = 8) at `v = c`,
    /// `u = b * c`, with the given step.
    pub fn new(dt: f64) -> Self {
        let c = -65.0;
        let b = 0.2;
        Self {
            v: c,
            u: b * c,
            a: 0.02,
            b,
            c,
            d: 8.0,
            dt,
        }
    }

    fn rhs(&self, v: f64, u: f64, current: f64) -> (f64, f64) {
        let dv = 0.04 * v.powi(2) + 5.0 * v + 140.0 - u + current;
        let du = self.a * (self.b * v - u);
        (dv, du)
    }

    /// Advance one RK4 step with the input held over it; return 1 on a spike, else 0.
    pub fn step(&mut self, current: f64) -> i32 {
        let (k1_v, k1_u) = self.rhs(self.v, self.u, current);
        let (k2_v, k2_u) = self.rhs(
            self.v + 0.5 * self.dt * k1_v,
            self.u + 0.5 * self.dt * k1_u,
            current,
        );
        let (k3_v, k3_u) = self.rhs(
            self.v + 0.5 * self.dt * k2_v,
            self.u + 0.5 * self.dt * k2_u,
            current,
        );
        let (k4_v, k4_u) = self.rhs(self.v + self.dt * k3_v, self.u + self.dt * k3_u, current);

        self.v += (self.dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        self.u += (self.dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u);

        if self.v >= IZH_SPIKE_THRESHOLD {
            self.v = self.c;
            self.u += self.d;
            1
        } else {
            0
        }
    }
}

/// Izhikevich 2007 biophysical parameterisation (NeuroML `izhikevich2007Cell`):
/// `C dv/dt = k (v - vr)(v - vt) - u + I`, `du/dt = a (b (v - vr) - u)`, with a
/// `v >= vpeak -> v = c, u += d` reset. RK4 over the coupled ODE. The right-hand
/// side is exact arithmetic (products, a sum, a division — no transcendental
/// functions), so `simulate` matches the Python reference bit-for-bit.
#[derive(Clone, Debug)]
pub struct Izhikevich2007Rk4 {
    /// Membrane potential (mV).
    pub v: f64,
    /// Recovery current (pA).
    pub u: f64,
    /// Membrane capacitance C (pF).
    pub cap: f64,
    /// Quadratic scaling of the spike-generating current (pA/mV^2).
    pub k: f64,
    /// Resting membrane potential (mV).
    pub vr: f64,
    /// Instantaneous threshold potential (mV).
    pub vt: f64,
    /// Spike cutoff (peak) potential; reaching it resets (mV).
    pub vpeak: f64,
    /// Recovery time scale (1/ms).
    pub a: f64,
    /// Sensitivity of the recovery variable to `v - vr` (nS).
    pub b: f64,
    /// Voltage reset value after a spike (mV).
    pub c: f64,
    /// Recovery-variable increment after a spike (pA).
    pub d: f64,
    /// RK4 step (ms).
    pub dt: f64,
}

impl Izhikevich2007Rk4 {
    /// Create the regular-spiking cell of Izhikevich (2007): C = 100 pF, k = 0.7, vr = -60 mV,
    /// vt = -40 mV, vpeak = 35 mV, a = 0.03, b = -2, c = -50 mV, d = 100, at `v = vr`,
    /// `u = 0`, with `dt` = 0.1 ms.
    pub fn new() -> Self {
        Self {
            v: -60.0,
            u: 0.0,
            cap: 100.0,
            k: 0.7,
            vr: -60.0,
            vt: -40.0,
            vpeak: 35.0,
            a: 0.03,
            b: -2.0,
            c: -50.0,
            d: 100.0,
            dt: 0.1,
        }
    }

    fn is_valid(&self) -> bool {
        [
            self.v, self.u, self.cap, self.k, self.vr, self.vt, self.vpeak, self.a, self.b, self.c,
            self.d, self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.cap > 0.0
            && self.dt > 0.0
    }

    fn rhs(&self, v: f64, u: f64, current: f64) -> (f64, f64) {
        let dv = (self.k * (v - self.vr) * (v - self.vt) - u + current) / self.cap;
        let du = self.a * (self.b * (v - self.vr) - u);
        (dv, du)
    }

    /// Advance one RK4 step with the input held over it; return 1 when `v` reaches `vpeak`.
    ///
    /// Reaching `vpeak` sets `v` to `c` and adds `d` to the updated `u`. A non-finite value,
    /// a `cap` or `dt` that is not positive, a non-finite input or a non-finite candidate is
    /// refused, and the state is left unchanged.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.is_valid() || !current.is_finite() {
            return Err("invalid Izhikevich 2007 runtime state or current");
        }
        let (k1v, k1u) = self.rhs(self.v, self.u, current);
        let (k2v, k2u) = self.rhs(
            self.v + 0.5 * self.dt * k1v,
            self.u + 0.5 * self.dt * k1u,
            current,
        );
        let (k3v, k3u) = self.rhs(
            self.v + 0.5 * self.dt * k2v,
            self.u + 0.5 * self.dt * k2u,
            current,
        );
        let (k4v, k4u) = self.rhs(self.v + self.dt * k3v, self.u + self.dt * k3u, current);
        let dt6 = self.dt / 6.0;
        let mut v_next = self.v + dt6 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        let mut u_next = self.u + dt6 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        let event = if v_next >= self.vpeak {
            v_next = self.c;
            u_next += self.d;
            1
        } else {
            0
        };
        if !v_next.is_finite() || !u_next.is_finite() {
            return Err("Izhikevich 2007 candidate state became non-finite");
        }
        self.v = v_next;
        self.u = u_next;
        Ok(event)
    }

    /// Compatibility update for the network runner; a refused step returns 0 and leaves the
    /// state unchanged.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run `n_steps` RK4 updates under a constant input, returning the `v` trace
    /// (already reset to `c` on spiking steps) and the spike count. Reuses
    /// `try_step`, so the trace is bit-identical to the per-step path and to the
    /// Python reference (the right-hand side is exact arithmetic). The final
    /// state is left in `self.v` / `self.u`. The first refused step returns its
    /// error, and the steps before it stay applied.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.try_step(current)?;
            trace.push(self.v);
            if spiked == 1 {
                spikes += 1;
            }
        }
        Ok((trace, spikes))
    }

    /// Return to `v = vr` and `u = b * (v - vr)`, which is zero; an invalid state is refused
    /// and left unchanged.
    pub fn try_reset(&mut self) -> Result<(), &'static str> {
        if !self.is_valid() {
            return Err("invalid Izhikevich 2007 parameters for reset");
        }
        let v_next = self.vr;
        let u_next = self.b * (v_next - self.vr);
        if !v_next.is_finite() || !u_next.is_finite() {
            return Err("Izhikevich 2007 reset state became non-finite");
        }
        self.v = v_next;
        self.u = u_next;
        Ok(())
    }

    /// Compatibility reset: [`Self::try_reset`], with a refusal ignored.
    pub fn reset(&mut self) {
        let _ = self.try_reset();
    }
}

impl Default for Izhikevich2007Rk4 {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive exponential integrate-and-fire (AdEx) neuron, one RK4 step of `dt` per call.
///
/// `v' = (-(v - v_rest) + delta_t exp((v - v_rh) / delta_t)) / tau + (-w + I) / c_m` and
/// `w' = (a (v - v_rest) - w) / tau_w`, the exponent clamped to `[-20, 20]` and the input
/// held over the step; `v >= v_threshold` after the step sets `v` to `v_reset` and adds `b`
/// to the updated `w`. Nothing is validated.
#[derive(Clone, Debug)]
pub struct AdExRk4 {
    /// Membrane potential (mV).
    pub v: f64,
    /// Adaptation current (pA).
    pub w: f64,
    /// Resting (leak reversal) potential (mV).
    pub v_rest: f64,
    /// Potential after a spike (mV).
    pub v_reset: f64,
    /// Spike cutoff; reaching it after a step resets (mV).
    pub v_threshold: f64,
    /// Rheobase threshold of the exponential term (mV).
    pub v_rh: f64,
    /// Slope factor of the exponential term (mV).
    pub delta_t: f64,
    /// Membrane time constant (ms).
    pub tau: f64,
    /// Adaptation time constant (ms).
    pub tau_w: f64,
    /// Subthreshold adaptation coupling (nS).
    pub a: f64,
    /// Spike-triggered adaptation increment (pA).
    pub b: f64,
    /// Membrane capacitance (pF).
    pub c_m: f64,
    /// RK4 step (ms).
    pub dt: f64,
}

impl AdExRk4 {
    /// Create a neuron at rest (`v` = -65 mV, `w` = 0) with the default parameters and the
    /// given step.
    pub fn new(dt: f64) -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -50.0,
            v_rh: -55.0,
            delta_t: 2.0,
            tau: 20.0,
            tau_w: 100.0,
            a: 0.5,
            b: 7.0,
            c_m: 200.0,
            dt,
        }
    }

    fn rhs(&self, v: f64, w: f64, current: f64) -> (f64, f64) {
        let exp_arg = ((v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = (-(v - self.v_rest) + exp_term) / self.tau + (-w + current) / self.c_m;
        let dw = (self.a * (v - self.v_rest) - w) / self.tau_w;
        (dv, dw)
    }

    /// Advance one RK4 step with the input held over it; return 1 on a spike, else 0.
    pub fn step(&mut self, current: f64) -> i32 {
        let (k1_v, k1_w) = self.rhs(self.v, self.w, current);
        let (k2_v, k2_w) = self.rhs(
            self.v + 0.5 * self.dt * k1_v,
            self.w + 0.5 * self.dt * k1_w,
            current,
        );
        let (k3_v, k3_w) = self.rhs(
            self.v + 0.5 * self.dt * k2_v,
            self.w + 0.5 * self.dt * k2_w,
            current,
        );
        let (k4_v, k4_w) = self.rhs(self.v + self.dt * k3_v, self.w + self.dt * k3_w, current);

        self.v += (self.dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        self.w += (self.dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w);

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.w += self.b;
            1
        } else {
            0
        }
    }
}

/// Hodgkin-Huxley squid-axon neuron, potentials shifted to a -65 mV rest, integrated with
/// classical RK4.
///
/// Each call takes `(1 / dt).round()` RK4 sub-steps of `dt`, about 1 ms, with a half rounded
/// away from zero; a `dt` above 2 gives no sub-step and leaves the state unchanged, and a `dt`
/// of exactly 2 takes one. A spike is an upward crossing of `v_threshold` between the start
/// and the end of the call. Nothing is validated.
#[derive(Clone, Debug)]
pub struct HodgkinHuxleyRk4 {
    /// Membrane potential (mV).
    pub v: f64,
    /// Sodium activation gate.
    pub m: f64,
    /// Sodium inactivation gate.
    pub h: f64,
    /// Potassium activation gate.
    pub n: f64,
    /// Membrane capacitance (uF/cm^2).
    pub c_m: f64,
    /// Maximal sodium conductance (mS/cm^2).
    pub g_na: f64,
    /// Maximal potassium conductance (mS/cm^2).
    pub g_k: f64,
    /// Leak conductance (mS/cm^2).
    pub g_l: f64,
    /// Sodium reversal potential (mV).
    pub e_na: f64,
    /// Potassium reversal potential (mV).
    pub e_k: f64,
    /// Leak reversal potential (mV).
    pub e_l: f64,
    /// RK4 sub-step (ms).
    pub dt: f64,
    /// Potential whose upward crossing counts as a spike (mV).
    pub v_threshold: f64,
}

impl HodgkinHuxleyRk4 {
    /// Create a neuron near rest (`v` = -65 mV, `m` = 0.05, `h` = 0.6, `n` = 0.32) with the
    /// standard conductances and reversal potentials, a spike threshold of 0 mV and the given
    /// sub-step.
    pub fn new(dt: f64) -> Self {
        Self {
            v: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.32,
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
            dt,
            v_threshold: 0.0,
        }
    }

    fn alpha_m(v: f64) -> f64 {
        let d = v + 40.0;
        if d.abs() < 1e-7 {
            1.0
        } else {
            0.1 * d / (1.0 - (-d / 10.0).exp())
        }
    }

    fn beta_m(v: f64) -> f64 {
        4.0 * (-(v + 65.0) / 18.0).exp()
    }

    fn alpha_h(v: f64) -> f64 {
        0.07 * (-(v + 65.0) / 20.0).exp()
    }

    fn beta_h(v: f64) -> f64 {
        1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
    }

    fn alpha_n(v: f64) -> f64 {
        let d = v + 55.0;
        if d.abs() < 1e-7 {
            0.1
        } else {
            0.01 * d / (1.0 - (-d / 10.0).exp())
        }
    }

    fn beta_n(v: f64) -> f64 {
        0.125 * (-(v + 65.0) / 80.0).exp()
    }

    fn rhs(&self, state: [f64; 4], current: f64) -> [f64; 4] {
        let [v, m, h, n] = state;
        let am = Self::alpha_m(v);
        let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v);
        let bh = Self::beta_h(v);
        let an = Self::alpha_n(v);
        let bn = Self::beta_n(v);

        let dm = am * (1.0 - m) - bm * m;
        let dh = ah * (1.0 - h) - bh * h;
        let dn = an * (1.0 - n) - bn * n;
        let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
        let i_k = self.g_k * n.powi(4) * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_na - i_k - i_l + current) / self.c_m;
        [dv, dm, dh, dn]
    }

    /// Advance about 1 ms (`(1 / dt).round()` sub-steps) with the input held; return 1 on an
    /// upward crossing of `v_threshold`, else 0.
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let mut state = [self.v, self.m, self.h, self.n];
        let substeps = (1.0 / self.dt).round() as usize;
        for _ in 0..substeps {
            let k1 = self.rhs(state, current);
            let k2 = self.rhs(add_scaled(state, k1, 0.5 * self.dt), current);
            let k3 = self.rhs(add_scaled(state, k2, 0.5 * self.dt), current);
            let k4 = self.rhs(add_scaled(state, k3, self.dt), current);
            for idx in 0..4 {
                state[idx] += (self.dt / 6.0) * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]);
            }
        }
        self.v = state[0];
        self.m = state[1];
        self.h = state[2];
        self.n = state[3];

        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
}

fn add_scaled(state: [f64; 4], deriv: [f64; 4], scale: f64) -> [f64; 4] {
    [
        state[0] + scale * deriv[0],
        state[1] + scale * deriv[1],
        state[2] + scale * deriv[2],
        state[3] + scale * deriv[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn izhikevich_rk4_is_deterministic_and_spikes() {
        let mut a = IzhikevichRk4::new(1.0);
        let mut b = IzhikevichRk4::new(1.0);
        let mut spikes = 0;
        for _ in 0..100 {
            spikes += a.step(10.0);
            b.step(10.0);
        }
        assert!(spikes > 0);
        assert_eq!(a.v, b.v);
        assert_eq!(a.u, b.u);
    }

    #[test]
    fn izhikevich2007_checked_trace_and_reset_are_fail_closed() {
        let mut neuron = Izhikevich2007Rk4::new();
        let (trace, events) = neuron.simulate(2_000, 100.0).unwrap();
        assert_eq!(trace.len(), 2_000);
        assert_eq!(events, 3);
        assert_eq!(trace.last().copied(), Some(neuron.v));

        let before = (neuron.v, neuron.u);
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!((neuron.v, neuron.u), before);

        neuron.vr = f64::NAN;
        assert!(neuron.try_reset().is_err());
        assert_eq!((neuron.v, neuron.u), before);
    }

    #[test]
    fn adex_rk4_remains_finite_under_sustained_current() {
        let mut neuron = AdExRk4::new(0.1);
        let mut spikes = 0;
        for _ in 0..3000 {
            spikes += neuron.step(500.0);
        }
        assert!(spikes > 0);
        assert!(neuron.v.is_finite());
        assert!(neuron.w.is_finite());
    }

    #[test]
    fn hodgkin_huxley_rk4_keeps_gates_bounded() {
        let mut neuron = HodgkinHuxleyRk4::new(0.01);
        let mut spikes = 0;
        for _ in 0..1000 {
            spikes += neuron.step(10.0);
        }
        assert!(spikes > 0);
        assert!(neuron.v.is_finite());
        assert!((0.0..=1.0).contains(&neuron.m));
        assert!((0.0..=1.0).contains(&neuron.h));
        assert!((0.0..=1.0).contains(&neuron.n));
    }
}
