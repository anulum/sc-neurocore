// SPDX-License-Identifier: AGPL-3.0-or-later
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

    pub fn default() -> Self {
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
        self.rate_trace = self.rate_trace * self.trace_decay + spike as f64 * (1.0 - self.trace_decay);

        // Threshold adaptation
        let error = self.rate_trace - self.target_rate;
        self.v_threshold += self.adaptation_rate * error;
        self.v_threshold = self
            .v_threshold
            .clamp(0.1, self.initial_threshold * 10.0);

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

    pub fn default() -> Self {
        Self::new(0.5)
    }

    pub fn step(&mut self, input_a: f64, input_b: f64) -> i32 {
        self.last_current = input_a + input_b - 2.0 * input_a * input_b;
        if self.last_current > self.threshold { 1 } else { 0 }
    }

    pub fn reset(&mut self) {
        self.last_current = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::{mask, BitstreamAverager, DendriticNeuron, FixedPointLif, HomeostaticLif, Izhikevich};

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
        let mut n = HomeostaticLif::default();
        let mut total = 0;
        for _ in 0..200 {
            total += n.step(25.0);
        }
        assert!(total > 0, "must fire with strong input");
    }

    #[test]
    fn homeostatic_threshold_adapts() {
        let mut n = HomeostaticLif::default();
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
        let mut n = HomeostaticLif::default();
        let mut total = 0;
        for _ in 0..100 {
            total += n.step(0.0);
        }
        assert_eq!(total, 0);
    }

    #[test]
    fn homeostatic_threshold_bounded() {
        let mut n = HomeostaticLif::default();
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
        let mut n = DendriticNeuron::default();
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
}
