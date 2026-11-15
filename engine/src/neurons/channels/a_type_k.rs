// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — A-type potassium channel neuron

#![deny(missing_docs)]

use crate::neurons::biophysical::safe_rate;

/// A-type K+ (IA) neuron — WB base + transient outward K+ current.
///
/// IA activates rapidly at subthreshold voltages and inactivates over
/// tens of milliseconds. The transient outward current opposes depolarisation,
/// creating a delay to the first spike and controlling interspike intervals.
///
/// Key mechanism for:
/// - First-spike latency: IA must inactivate before a spike can occur
/// - Spike frequency control: IA recovery during ISI limits firing rate
/// - Coincidence detection: neurons with strong IA prefer synchronous input
/// - Dendritic signal processing (hippocampal CA1 dendrites)
///
/// Connor & Stevens, J Physiol 213:31, 1971; Hoffman et al., Nature 387:869, 1997.
#[derive(Clone, Debug)]
pub struct ATypeKNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Sodium inactivation gate (dimensionless); its kinetics are scaled by `phi`.
    pub h: f64,
    /// Delayed-rectifier potassium activation gate (dimensionless); its kinetics are scaled
    /// by `phi`.
    pub n: f64,
    /// IA activation (fast).
    pub a: f64,
    /// IA inactivation (slow).
    pub b: f64,
    /// Maximal sodium conductance (mS/cm^2).
    pub g_na: f64,
    /// Maximal delayed-rectifier potassium conductance (mS/cm^2).
    pub g_k: f64,
    /// A-type transient potassium conductance (mS/cm^2).
    pub g_a: f64,
    /// Leak conductance (mS/cm^2).
    pub g_l: f64,
    /// Sodium reversal potential (mV).
    pub e_na: f64,
    /// Potassium reversal potential (mV).
    pub e_k: f64,
    /// Leak reversal potential (mV).
    pub e_l: f64,
    /// Membrane capacitance (uF/cm^2).
    pub c_m: f64,
    /// Temperature scaling factor for gating kinetics (dimensionless).
    pub phi: f64,
    /// Outer step (ms); `step` splits it into 50 forward-Euler sub-steps.
    pub dt: f64,
    /// Spike-detection threshold (mV).
    pub v_threshold: f64,
    /// Gain applied to the injected drive (dimensionless).
    pub gain: f64,
}

impl Default for ATypeKNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl ATypeKNeuron {
    /// Create a neuron at the descriptor's initial state with its default parameters and
    /// `dt` = 0.5 ms.
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            a: 0.1,
            b: 0.8,
            g_na: 35.0,
            g_k: 9.0,
            g_a: 8.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
        }
    }

    /// Advance `dt` in 50 forward-Euler sub-steps with the drive `gain * current`; return 1
    /// if any sub-step spiked, else 0.
    ///
    /// Each sub-step first updates the gates `h`, `n`, `a` and `b` from the old `v`. It then
    /// computes the currents with the updated gates (sodium activation at its steady state;
    /// A-current `g_a * a^3 * b * (v - e_k)`) and advances `v`. A sub-step that reaches
    /// `v_threshold` sets `v` to -65 mV and keeps the gates.
    ///
    /// After the sub-steps `v` is clamped to `[-100, 60]` mV. A non-finite `v` restores `v`,
    /// `h` and `n` to their initial values. The four gates are clamped to `[0, 1]` (a NaN
    /// gate stays NaN). The input is not validated. The Python model refuses a candidate
    /// outside `[-100, 60]` mV instead of clamping it.
    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let sub_steps = 50;
        let sub_dt = self.dt / sub_steps as f64;
        let mut fired = 0i32;

        for _ in 0..sub_steps {
            let v = self.v;

            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // A-type K+ gating (Connor-Stevens)
            let a_inf = 1.0 / (1.0 + (-(v + 50.0) / 20.0).exp());
            let tau_a = 2.0;
            let b_inf = 1.0 / (1.0 + ((v + 70.0) / 6.0).exp());
            let tau_b = 50.0;

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);
            self.a += sub_dt * (a_inf - self.a) / tau_a;
            self.b += sub_dt * (b_inf - self.b) / tau_b;

            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_a = self.g_a * self.a.powi(3) * self.b * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_a - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
            }
        }

        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -65.0;
            self.h = 0.6;
            self.n = 0.32;
        }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
        self.a = self.a.clamp(0.0, 1.0);
        self.b = self.b.clamp(0.0, 1.0);

        fired
    }

    /// Restore every field to `new()`, parameters included. The Python model's reset
    /// restores only the state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neurons::host_budget;

    // -- A-type K+ Neuron tests --

    #[test]
    fn atype_fires_with_input() {
        let mut n = ATypeKNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(3.0);
        }
        assert!(
            spikes > 5,
            "A-type neuron must fire with input, got {spikes}"
        );
    }

    #[test]
    fn atype_silent_without_input() {
        let mut n = ATypeKNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "A-type neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn atype_delays_first_spike() {
        // IA creates onset delay — removing IA should shorten latency
        let mut with_ia = ATypeKNeuron::new();
        let mut no_ia = ATypeKNeuron::new();
        no_ia.g_a = 0.0;

        let input = 3.0;
        let mut time_with = 10_000usize;
        for i in 0..10_000 {
            if with_ia.step(input) > 0 {
                time_with = i;
                break;
            }
        }
        let mut time_no = 10_000usize;
        for i in 0..10_000 {
            if no_ia.step(input) > 0 {
                time_no = i;
                break;
            }
        }
        assert!(
            time_with >= time_no,
            "IA must delay first spike: with={time_with} vs without={time_no}"
        );
    }

    #[test]
    fn atype_reduces_firing_rate() {
        // IA should reduce steady-state firing rate
        let mut with_ia = ATypeKNeuron::new();
        let mut no_ia = ATypeKNeuron::new();
        no_ia.g_a = 0.0;

        let input = 3.0;
        let mut spikes_ia = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_ia += with_ia.step(input);
            spikes_no += no_ia.step(input);
        }
        assert!(
            spikes_no >= spikes_ia,
            "IA should reduce firing rate: IA={spikes_ia} vs none={spikes_no}"
        );
    }

    #[test]
    fn atype_negative_input_no_crash() {
        let mut n = ATypeKNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn atype_nan_input_stays_finite() {
        let mut n = ATypeKNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn atype_extreme_input_bounded() {
        let mut n = ATypeKNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn atype_reset_clears_state() {
        let mut n = ATypeKNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.a, 0.1);
        assert_eq!(n.b, 0.8);
    }

    #[test]
    fn atype_gates_bounded() {
        let mut n = ATypeKNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        assert!(n.a >= 0.0 && n.a <= 1.0);
        assert!(n.b >= 0.0 && n.b <= 1.0);
    }

    #[test]
    fn atype_performance_1k_steps() {
        host_budget::assert_within(module_path!(), "atype_performance_1k_steps", || {
            let mut n = ATypeKNeuron::new();
            for _ in 0..1_000 {
                std::hint::black_box(n.step(3.0));
            }
        });
    }
}
