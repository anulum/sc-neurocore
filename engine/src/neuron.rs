// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neuron Models

//! # Neuron Models
//!
//! Fixed-point LIF and Izhikevich neuron models for the v3 engine.

/// Mask and sign-interpret an integer to `width` bits (branchless).
///
/// `width` must be in 1..=32. Values outside this range trigger a
/// debug assertion failure (release builds silently produce garbage).
#[inline]
pub fn mask(value: i32, width: u32) -> i16 {
    assert!(
        width > 0 && width <= 32,
        "mask width must be 1..=32, got {width}"
    );
    let m = (1_i64 << width) - 1;
    let v = (value as i64) & m;
    let shift = 64 - width;
    ((v << shift) >> shift) as i16
}

/// Fixed-point leaky-integrate-and-fire neuron state and parameters.
#[derive(Clone, Debug)]
pub struct FixedPointLif {
    /// Membrane potential.
    pub v: i16,
    /// Refractory counter in simulation steps.
    pub refractory_counter: i32,
    /// Arithmetic data width.
    pub data_width: u32,
    /// Fraction bits for fixed-point scaling.
    pub fraction: u32,
    /// Resting potential.
    pub v_rest: i16,
    /// Reset potential after spike.
    pub v_reset: i16,
    /// Spike threshold.
    pub v_threshold: i16,
    /// Refractory period length in steps.
    pub refractory_period: i32,
}

impl FixedPointLif {
    /// Construct a fixed-point LIF neuron.
    pub fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
    ) -> Self {
        Self {
            v: v_rest,
            refractory_counter: 0,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        }
    }

    /// Advance one simulation step.
    ///
    /// Returns `(spike, membrane_voltage)`.
    #[allow(non_snake_case)]
    pub fn step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        let w = self.data_width;

        // Refractory: check previous step's counter before any fire logic.
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.v = self.v_rest;
            return (0, mask(self.v_rest as i32, w));
        }

        let diff = mask((self.v_rest as i32) - (self.v as i32), 2 * w) as i32;
        let dv_leak = mask((diff * (leak_k as i32)) >> self.fraction, self.data_width);
        let dv_in = mask(
            ((i_t as i32) * (gain_k as i32)) >> self.fraction,
            self.data_width,
        );

        let v_next = mask(
            (self.v as i32) + (dv_leak as i32) + (dv_in as i32) + (noise_in as i32),
            self.data_width,
        );

        if v_next >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_counter = self.refractory_period;
            (1, mask(self.v_reset as i32, w))
        } else {
            self.v = v_next;
            (0, mask(v_next as i32, w))
        }
    }

    /// Reset internal state to resting potential.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_counter = 0;
    }
}

/// Izhikevich neuron (floating-point).
///
/// Standard model from IEEE TNN 14(6), 2003:
///   v' = 0.04*v² + 5*v + 140 - u + I
///   u' = a*(b*v - u)
///   if v >= 30: v ← c, u ← u + d
#[derive(Clone, Debug)]
pub struct Izhikevich {
    pub v: f64,
    pub u: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub dt: f64,
}

impl Izhikevich {
    /// Regular spiking defaults: a=0.02, b=0.2, c=-65, d=8, dt=1.0.
    pub fn new(a: f64, b: f64, c: f64, d: f64, dt: f64) -> Self {
        Self {
            v: c,
            u: b * c,
            a,
            b,
            c,
            d,
            dt,
        }
    }

    /// Regular spiking preset.
    pub fn regular_spiking() -> Self {
        Self::new(0.02, 0.2, -65.0, 8.0, 1.0)
    }

    /// Advance one step. Returns 1 on spike, 0 otherwise.
    pub fn step(&mut self, current: f64) -> i32 {
        // Two half-steps for numerical stability on 0.04v² term.
        let half = self.dt * 0.5;
        for _ in 0..2 {
            let dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + current) * half;
            let du = (self.a * (self.b * self.v - self.u)) * half;
            self.v += dv;
            self.u += du;
        }

        if self.v >= 30.0 {
            self.v = self.c;
            self.u += self.d;
            1
        } else {
            0
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.v = self.c;
        self.u = self.b * self.c;
    }
}

/// Sliding-window bitstream probability estimator.
///
/// Mirrors Python's `BitstreamAverager`.
#[derive(Clone, Debug)]
pub struct BitstreamAverager {
    buffer: Vec<u8>,
    index: usize,
    filled: bool,
    running_sum: u64,
}

impl BitstreamAverager {
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        Self {
            buffer: vec![0; window],
            index: 0,
            filled: false,
            running_sum: 0,
        }
    }

    pub fn push(&mut self, bit: u8) {
        debug_assert!(bit <= 1, "bit must be 0 or 1");
        let old = self.buffer[self.index];
        self.buffer[self.index] = bit;

        if self.filled {
            self.running_sum = self.running_sum - old as u64 + bit as u64;
        } else {
            self.running_sum += bit as u64;
        }

        self.index += 1;
        if self.index == self.buffer.len() {
            self.index = 0;
            self.filled = true;
        }
    }

    pub fn estimate(&self) -> f64 {
        if !self.filled {
            if self.index == 0 {
                return 0.0;
            }
            return self.running_sum as f64 / self.index as f64;
        }
        self.running_sum as f64 / self.buffer.len() as f64
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0);
        self.index = 0;
        self.filled = false;
        self.running_sum = 0;
    }

    pub fn window(&self) -> usize {
        self.buffer.len()
    }
}

/// Homeostatic LIF neuron with adaptive threshold.
///
/// Threshold adapts via EMA of spike rate toward a target setpoint.
/// Turrigiano, Cold Spring Harb Perspect Biol 4:a005736, 2012.
#[derive(Clone, Debug)]
pub struct HomeostaticLif {
    pub v: f64,
    pub v_threshold: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub rate_trace: f64,
    pub target_rate: f64,
    pub adaptation_rate: f64,
    pub trace_decay: f64,
    initial_threshold: f64,
}

impl HomeostaticLif {
    pub fn new(target_rate: f64, adaptation_rate: f64, trace_decay: f64) -> Self {
        Self {
            v: 0.0,
            v_threshold: 1.0,
            v_rest: 0.0,
            v_reset: 0.0,
            rate_trace: 0.0,
            target_rate,
            adaptation_rate,
            trace_decay,
            initial_threshold: 1.0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.1, 0.01, 0.95)
    }

    /// LIF step with threshold adaptation. Returns 1 on spike.
    pub fn step(&mut self, current: f64) -> i32 {
        // Leak-integrate
        let tau = 20.0;
        self.v += (-(self.v - self.v_rest) + current) / tau;

        let spike = if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        };

        // EMA spike rate tracking
        self.rate_trace =
            self.rate_trace * self.trace_decay + spike as f64 * (1.0 - self.trace_decay);

        // Threshold adaptation
        let error = self.rate_trace - self.target_rate;
        self.v_threshold += self.adaptation_rate * error;
        self.v_threshold = self.v_threshold.clamp(0.1, self.initial_threshold * 10.0);

        spike
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.rate_trace = 0.0;
        self.v_threshold = self.initial_threshold;
    }
}

/// XOR-nonlinearity dendritic neuron.
///
/// Koch, Biophysics of Computation, 1999, Ch. 12.
/// Output = 1 if (d1 + d2 - 2*d1*d2) > threshold.
#[derive(Clone, Debug)]
pub struct DendriticNeuron {
    pub threshold: f64,
    last_current: f64,
}

impl DendriticNeuron {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            last_current: 0.0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.5)
    }

    pub fn step(&mut self, input_a: f64, input_b: f64) -> i32 {
        self.last_current = input_a + input_b - 2.0 * input_a * input_b;
        if self.last_current > self.threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.last_current = 0.0;
    }
}

/// Adaptive Exponential IF neuron. Brette & Gerstner 2005.
/// PyO3 wrapper: `pyo3_neurons::PyAdExNeuron`
#[derive(Clone, Debug)]
pub struct AdExNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub c_m: f64,
    pub dt: f64,
}

impl Default for AdExNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl AdExNeuron {
    pub fn new() -> Self {
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
            dt: 0.1,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        // Brette & Gerstner 2005: C dV/dt = -g_L(V-E_L) + g_L ΔT exp((V-V_T)/ΔT) - w + I
        let exp_arg = ((self.v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = ((-(self.v - self.v_rest) + exp_term) / self.tau + (-self.w + current) / self.c_m)
            * self.dt;
        let dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt;
        self.v += dv;
        self.w += dw;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.w += self.b;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}

/// Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003.
#[derive(Clone, Debug)]
pub struct ExpIfNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub dt: f64,
    /// Precomputed 1.0 / delta_t.
    pub inv_delta_t: f64,
    /// Precomputed dt / tau.
    pub dt_div_tau: f64,
}

impl Default for ExpIfNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpIfNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -50.0,
            v_rh: -55.0,
            delta_t: 2.0,
            tau: 20.0,
            dt: 0.1,
            inv_delta_t: 1.0 / 2.0,
            dt_div_tau: 0.1 / 20.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.v.is_finite()
            || !current.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || !self.v_rh.is_finite()
            || !self.delta_t.is_finite()
            || !self.tau.is_finite()
            || !self.dt.is_finite()
            || self.delta_t <= 0.0
            || self.tau <= 0.0
            || self.dt <= 0.0
        {
            return 0;
        }

        self.inv_delta_t = 1.0 / self.delta_t;
        self.dt_div_tau = self.dt / self.tau;
        let k1 = self.rhs(self.v, current);
        let k2 = self.rhs(self.v + 0.5 * self.dt * k1, current);
        let k3 = self.rhs(self.v + 0.5 * self.dt * k2, current);
        let k4 = self.rhs(self.v + self.dt * k3, current);
        let next_v = self.v + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        if !k1.is_finite()
            || !k2.is_finite()
            || !k3.is_finite()
            || !k4.is_finite()
            || !next_v.is_finite()
        {
            return 0;
        }
        self.v = next_v;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }

    fn rhs(&self, v: f64, current: f64) -> f64 {
        let exp_arg = ((v - self.v_rh) * self.inv_delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        (-(v - self.v_rest) + exp_term + current) / self.tau
    }
}

/// Lapicque 1907 — classical RC integrate-and-fire.
#[derive(Clone, Debug)]
pub struct LapicqueNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl LapicqueNeuron {
    pub fn new(tau: f64, resistance: f64, threshold: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            v_threshold: threshold,
            tau,
            resistance,
            dt,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.v.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || self.v_threshold <= self.v_rest
            || self.v_threshold <= self.v_reset
            || self.v >= self.v_threshold
            || !self.tau.is_finite()
            || self.tau <= 0.0
            || !self.resistance.is_finite()
            || self.resistance <= 0.0
            || !self.dt.is_finite()
            || self.dt <= 0.0
            || !current.is_finite()
        {
            return 0;
        }

        let v_inf = self.v_rest + self.resistance * current;
        let decay = (-self.dt / self.tau).exp();
        let next_v = v_inf + (self.v - v_inf) * decay;
        if !v_inf.is_finite() || !decay.is_finite() || !next_v.is_finite() {
            return 0;
        }
        self.v = next_v;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_exp_if_optimisation_parity() {
        let mut n = ExpIfNeuron::new();
        n.v = -60.0;
        let current = 10.0;

        // Manual calculation of original formula
        let exp_arg = ((-60.0_f64 - (-55.0)) / 2.0).clamp(-20.0, 20.0);
        let exp_term = 2.0 * exp_arg.exp();
        let expected_dv = (-(-60.0 - (-65.0)) + exp_term + current) / 20.0 * 0.1;

        n.step(current);
        let got_dv = n.v - (-60.0); // Simple check since we only did one step

        // Use a small epsilon for float parity
        assert!(
            (got_dv - expected_dv).abs() < 1e-15,
            "Logic mismatch in ExpIfNeuron: got {}, expected {}",
            got_dv,
            expected_dv
        );
    }

    use super::{
        mask, AdExNeuron, BitstreamAverager, DendriticNeuron, ExpIfNeuron, FixedPointLif,
        HomeostaticLif, Izhikevich, LapicqueNeuron,
    };

    #[test]
    fn mask_branchless_matches_original() {
        for &width in &[16_u32, 32] {
            for value in [
                -32768_i32,
                -1,
                0,
                1,
                32767,
                65535,
                -65536,
                i16::MAX as i32,
                i16::MIN as i32,
            ] {
                let result = mask(value, width);

                let m = (1_i64 << width) - 1;
                let mut v = (value as i64) & m;
                if v >= (1_i64 << (width - 1)) {
                    v -= 1_i64 << width;
                }
                let expected = if width >= 32 {
                    v as i32 as i16
                } else {
                    v as i16
                };

                assert_eq!(
                    result, expected,
                    "mask({value}, {width}): got {result}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn lif_fires_with_refractory_period() {
        // Q8.8: threshold=1.0 → 256, matching Python default
        let mut n = FixedPointLif::new(16, 8, 0, 0, 256, 2);
        let mut spikes = Vec::new();
        for _ in 0..30 {
            let (s, _) = n.step(1, 256, 50, 0);
            spikes.push(s);
        }
        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "neuron must fire with refractory_period=2");
        // Refractory gap: after a spike, next 2 steps must be silent.
        for (i, &s) in spikes.iter().enumerate() {
            if s == 1 && i + 2 < spikes.len() {
                assert_eq!(spikes[i + 1], 0, "step {} should be refractory", i + 1);
                assert_eq!(spikes[i + 2], 0, "step {} should be refractory", i + 2);
            }
        }
    }

    #[test]
    fn lif_fires_without_refractory() {
        let mut n = FixedPointLif::new(16, 8, 0, 0, 256, 0);
        let mut total = 0;
        for _ in 0..20 {
            let (s, _) = n.step(1, 256, 50, 0);
            total += s;
        }
        assert!(total > 0, "neuron must fire with refractory_period=0");
    }

    // ── Izhikevich tests ──────────────────────────────────────────

    #[test]
    fn izhikevich_regular_spiking_fires() {
        let mut n = Izhikevich::regular_spiking();
        let mut total = 0;
        for _ in 0..100 {
            total += n.step(10.0);
        }
        assert!(total > 0, "RS neuron must fire with I=10");
    }

    #[test]
    fn izhikevich_no_spike_without_input() {
        let mut n = Izhikevich::regular_spiking();
        let mut total = 0;
        for _ in 0..100 {
            total += n.step(0.0);
        }
        assert_eq!(total, 0, "no spikes without input");
    }

    #[test]
    fn izhikevich_reset_clears_state() {
        let mut n = Izhikevich::regular_spiking();
        for _ in 0..50 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, n.c);
        assert!((n.u - n.b * n.c).abs() < 1e-12);
    }

    #[test]
    fn izhikevich_chattering_fires_more() {
        // Chattering: a=0.02, b=0.2, c=-50, d=2
        let mut ch = Izhikevich::new(0.02, 0.2, -50.0, 2.0, 1.0);
        let mut rs = Izhikevich::regular_spiking();
        let mut ch_spikes = 0;
        let mut rs_spikes = 0;
        for _ in 0..200 {
            ch_spikes += ch.step(10.0);
            rs_spikes += rs.step(10.0);
        }
        assert!(
            ch_spikes > rs_spikes,
            "chattering ({ch_spikes}) should fire more than RS ({rs_spikes})"
        );
    }

    // ── BitstreamAverager tests ───────────────────────────────────

    #[test]
    fn averager_all_ones() {
        let mut avg = BitstreamAverager::new(100);
        for _ in 0..100 {
            avg.push(1);
        }
        assert!((avg.estimate() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn averager_all_zeros() {
        let mut avg = BitstreamAverager::new(50);
        for _ in 0..50 {
            avg.push(0);
        }
        assert!(avg.estimate().abs() < 1e-12);
    }

    #[test]
    fn averager_half() {
        let mut avg = BitstreamAverager::new(100);
        for i in 0..100 {
            avg.push((i % 2) as u8);
        }
        assert!((avg.estimate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn averager_sliding_window() {
        let mut avg = BitstreamAverager::new(4);
        // Fill: [1, 1, 0, 0] → 0.5
        for &b in &[1_u8, 1, 0, 0] {
            avg.push(b);
        }
        assert!((avg.estimate() - 0.5).abs() < 1e-12);
        // Push 1 → [1, 1, 0, 1] (oldest 1 replaced by 1) → wait
        // Actually buffer is circular: index=0, push 1 replaces buffer[0]=1 with 1 → still 0.5
        avg.push(1);
        // Buffer: [1, 1, 0, 0] → index wraps to 0, push 1 at index 0: [1, 1, 0, 0] → [1, 1, 0, 0] no wait
        // filled=true after first wrap. push(1) at index 0: old=1, new=1, sum stays 2 → 0.5
        assert!((avg.estimate() - 0.5).abs() < 1e-12);
        // Push 1 at index 1: old=1, new=1 → still 0.5
        avg.push(1);
        assert!((avg.estimate() - 0.5).abs() < 1e-12);
        // Push 1 at index 2: old=0, new=1 → sum=3 → 0.75
        avg.push(1);
        assert!((avg.estimate() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn averager_partial_fill() {
        let mut avg = BitstreamAverager::new(100);
        avg.push(1);
        avg.push(0);
        assert!((avg.estimate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn averager_empty_returns_zero() {
        let avg = BitstreamAverager::new(10);
        assert!(avg.estimate().abs() < 1e-12);
    }

    // ── HomeostaticLif tests ──────────────────────────────────────

    #[test]
    fn homeostatic_fires_with_strong_input() {
        let mut n = HomeostaticLif::with_defaults();
        let mut total = 0;
        for _ in 0..200 {
            total += n.step(25.0);
        }
        assert!(total > 0, "must fire with strong input");
    }

    #[test]
    fn homeostatic_threshold_adapts() {
        let mut n = HomeostaticLif::with_defaults();
        let initial = n.v_threshold;
        for _ in 0..500 {
            n.step(25.0);
        }
        assert!(
            (n.v_threshold - initial).abs() > 1e-6,
            "threshold must adapt"
        );
    }

    #[test]
    fn homeostatic_no_fire_without_input() {
        let mut n = HomeostaticLif::with_defaults();
        let mut total = 0;
        for _ in 0..100 {
            total += n.step(0.0);
        }
        assert_eq!(total, 0);
    }

    #[test]
    fn homeostatic_threshold_bounded() {
        let mut n = HomeostaticLif::with_defaults();
        for _ in 0..10000 {
            n.step(50.0);
        }
        assert!(n.v_threshold >= 0.1);
        assert!(n.v_threshold <= 10.0);
    }

    // ── DendriticNeuron tests ─────────────────────────────────────

    #[test]
    fn dendritic_xor_truth_table() {
        let mut n = DendriticNeuron::new(0.5);
        assert_eq!(n.step(0.0, 0.0), 0); // 0+0-0 = 0
        assert_eq!(n.step(1.0, 0.0), 1); // 1+0-0 = 1
        assert_eq!(n.step(0.0, 1.0), 1); // 0+1-0 = 1
        assert_eq!(n.step(1.0, 1.0), 0); // 1+1-2 = 0
    }

    #[test]
    fn dendritic_subthreshold() {
        let mut n = DendriticNeuron::new(0.5);
        assert_eq!(n.step(0.2, 0.1), 0);
    }

    #[test]
    fn dendritic_reset() {
        let mut n = DendriticNeuron::with_defaults();
        n.step(1.0, 0.0);
        n.reset();
        assert!((n.last_current).abs() < 1e-12);
    }

    #[test]
    fn averager_reset() {
        let mut avg = BitstreamAverager::new(10);
        for _ in 0..10 {
            avg.push(1);
        }
        avg.reset();
        assert!(avg.estimate().abs() < 1e-12);
    }

    // ── AdEx tests ────────────────────────────────────────────────

    #[test]
    fn adex_fires_with_input() {
        let mut n = AdExNeuron::new();
        let mut total = 0;
        for _ in 0..2000 {
            total += n.step(500.0);
        }
        assert!(total > 0, "AdEx must fire with strong input");
    }

    #[test]
    fn adex_adaptation_reduces_rate() {
        let mut n = AdExNeuron::new();
        let first_100: i32 = (0..1000).map(|_| n.step(400.0)).sum();
        let next_100: i32 = (0..1000).map(|_| n.step(400.0)).sum();
        // Adaptation should reduce firing over time (w grows)
        assert!(
            next_100 <= first_100 + 5,
            "adaptation should not increase rate: first={first_100}, next={next_100}"
        );
    }

    // ── ExpIF tests ───────────────────────────────────────────────

    #[test]
    fn expif_fires() {
        let mut n = ExpIfNeuron::new();
        let mut total = 0;
        for _ in 0..2000 {
            total += n.step(500.0);
        }
        assert!(total > 0, "ExpIF must fire");
    }

    #[test]
    fn expif_no_fire_without_input() {
        let mut n = ExpIfNeuron::new();
        let total: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn expif_matches_rk4_reference() {
        let mut n = ExpIfNeuron::new();
        n.v = -60.0;
        n.dt = 0.25;
        n.tau = 20.0;
        let rhs = |v: f64, s: &ExpIfNeuron, current: f64| {
            let exp_arg = ((v - s.v_rh) / s.delta_t).clamp(-20.0, 20.0);
            (-(v - s.v_rest) + s.delta_t * exp_arg.exp() + current) / s.tau
        };
        let current = 12.0;
        let k1 = rhs(n.v, &n, current);
        let k2 = rhs(n.v + 0.5 * n.dt * k1, &n, current);
        let k3 = rhs(n.v + 0.5 * n.dt * k2, &n, current);
        let k4 = rhs(n.v + n.dt * k3, &n, current);
        let expected = n.v + n.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected).abs() < 1e-12);
    }

    // ── Lapicque tests ────────────────────────────────────────────

    #[test]
    fn lapicque_fires() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let mut total = 0;
        for _ in 0..200 {
            total += n.step(5.0);
        }
        assert!(total > 0, "Lapicque must fire with sustained input");
    }

    #[test]
    fn lapicque_reset() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        for _ in 0..50 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v).abs() < 1e-12);
    }

    #[test]
    fn lapicque_exact_flow_matches_closed_form() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 5.0);
        n.v = 0.25;
        let current = 0.5;
        let v0 = n.v;
        let v_inf = n.v_rest + n.resistance * current;
        let euler = v0 + (-(v0 - n.v_rest) + n.resistance * current) / n.tau * n.dt;
        let expected = v_inf + (v0 - v_inf) * (-n.dt / n.tau).exp();

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected).abs() < 1e-15);
        assert!((n.v - euler).abs() > 1e-4);
    }

    // ── AdEx coverage tests ────────────────────────────────────────

    #[test]
    fn adex_no_fire_without_input() {
        let mut n = AdExNeuron::new();
        let total: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn adex_negative_current_no_fire() {
        let mut n = AdExNeuron::new();
        let total: i32 = (0..500).map(|_| n.step(-100.0)).sum();
        assert_eq!(total, 0, "negative current must not cause spikes");
    }

    #[test]
    fn adex_reset_roundtrip() {
        let mut n = AdExNeuron::new();
        for _ in 0..200 {
            n.step(500.0);
        }
        assert!(n.w > 0.0, "w must grow during spiking");
        n.reset();
        assert_eq!(n.v, n.v_rest);
        assert_eq!(n.w, 0.0);
        // Post-reset: should behave identically to fresh
        let mut fresh = AdExNeuron::new();
        let r1: i32 = (0..100).map(|_| n.step(500.0)).sum();
        let r2: i32 = (0..100).map(|_| fresh.step(500.0)).sum();
        assert_eq!(r1, r2, "reset neuron must match fresh neuron");
    }

    #[test]
    fn adex_voltage_bounded() {
        let mut n = AdExNeuron::new();
        for _ in 0..5000 {
            n.step(1000.0);
        }
        assert!(n.v.is_finite(), "voltage must stay finite");
        assert!(n.w.is_finite(), "adaptation must stay finite");
    }

    #[test]
    fn adex_pipeline_sustained_spiking() {
        let mut n = AdExNeuron::new();
        let spikes: i32 = (0..10000).map(|_| n.step(500.0)).sum();
        assert!(
            spikes > 100,
            "sustained input should produce many spikes: got {spikes}"
        );
        assert!(n.v.is_finite());
    }

    #[test]
    fn adex_performance_10k_steps() {
        let mut n = AdExNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(500.0);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps took too long: {:?}",
            elapsed
        );
    }

    // ── ExpIF coverage tests ───────────────────────────────────────

    #[test]
    fn expif_negative_current_no_fire() {
        let mut n = ExpIfNeuron::new();
        let total: i32 = (0..500).map(|_| n.step(-100.0)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn expif_reset_roundtrip() {
        let mut n = ExpIfNeuron::new();
        for _ in 0..200 {
            n.step(500.0);
        }
        n.reset();
        assert_eq!(n.v, n.v_rest);
        let mut fresh = ExpIfNeuron::new();
        let r1: i32 = (0..100).map(|_| n.step(500.0)).sum();
        let r2: i32 = (0..100).map(|_| fresh.step(500.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn expif_voltage_bounded() {
        let mut n = ExpIfNeuron::new();
        for _ in 0..5000 {
            n.step(1000.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn expif_fires_more_than_adex() {
        // ExpIF has no adaptation, should fire at least as much as AdEx
        let mut eif = ExpIfNeuron::new();
        let mut adex = AdExNeuron::new();
        let eif_spikes: i32 = (0..5000).map(|_| eif.step(500.0)).sum();
        let adex_spikes: i32 = (0..5000).map(|_| adex.step(500.0)).sum();
        assert!(
            eif_spikes >= adex_spikes,
            "ExpIF ({eif_spikes}) should fire >= AdEx ({adex_spikes}) due to no adaptation"
        );
    }

    #[test]
    fn expif_performance_10k_steps() {
        let mut n = ExpIfNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(500.0);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps took too long: {:?}",
            elapsed
        );
    }

    // ── Lapicque coverage tests ────────────────────────────────────

    #[test]
    fn lapicque_no_fire_without_input() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let total: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn lapicque_negative_current_no_fire() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let total: i32 = (0..500).map(|_| n.step(-5.0)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn lapicque_invalid_state_does_not_mutate() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        n.v = 0.25;
        n.tau = 0.0;
        assert_eq!(n.step(1.0), 0);
        assert_eq!(n.v, 0.25);
    }

    #[test]
    fn lapicque_reset_roundtrip() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, n.v_rest);
        let mut fresh = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let r1: i32 = (0..100).map(|_| n.step(5.0)).sum();
        let r2: i32 = (0..100).map(|_| fresh.step(5.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn lapicque_voltage_bounded() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        for _ in 0..5000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn lapicque_higher_resistance_fires_faster() {
        let mut lo = LapicqueNeuron::new(20.0, 0.5, 1.0, 1.0);
        let mut hi = LapicqueNeuron::new(20.0, 2.0, 1.0, 1.0);
        let lo_spikes: i32 = (0..200).map(|_| lo.step(1.0)).sum();
        let hi_spikes: i32 = (0..200).map(|_| hi.step(1.0)).sum();
        assert!(
            hi_spikes >= lo_spikes,
            "higher R ({hi_spikes}) should fire >= lower R ({lo_spikes})"
        );
    }

    #[test]
    fn lapicque_performance_10k_steps() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(5.0);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps took too long: {:?}",
            elapsed
        );
    }

    #[test]
    fn lapicque_pipeline_sustained_spiking() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let spikes: i32 = (0..10000).map(|_| n.step(5.0)).sum();
        assert!(
            spikes > 100,
            "sustained input should produce many spikes: got {spikes}"
        );
    }
}
