// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Chandelier interneuron model

#![deny(missing_docs)]

use super::super::biophysical::safe_rate;

/// Chandelier cell — axo-axonic fast-spiking interneuron.
///
/// Biophysics: Wang-Buzsáki core + Kv1 (D-type delay current) + Kv3.1.
/// Kv1 creates a delay to first spike compared to PV+. Targets AIS.
///
/// Based on Woodruff et al. 2011 / Wang & Buzsáki 1996.
#[derive(Clone, Debug)]
pub struct ChandelierNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Sodium inactivation gate (dimensionless); Wang-Buzsáki kinetics scaled by `phi`.
    pub h: f64,
    /// Delayed-rectifier potassium activation gate (dimensionless); kinetics scaled by
    /// `phi`.
    pub n: f64,
    /// Kv1 (D-type) activation gate (dimensionless).
    pub d: f64,
    /// Kv3.1 activation gate (dimensionless).
    pub p: f64,
    // Conductances
    /// Maximal sodium conductance (mS/cm^2).
    pub g_na: f64,
    /// Maximal delayed-rectifier potassium conductance (mS/cm^2).
    pub g_k: f64,
    /// Kv1 low-threshold potassium conductance (mS/cm^2).
    pub g_kv1: f64,
    /// Kv3 fast delayed-rectifier potassium conductance (mS/cm^2).
    pub g_kv3: f64,
    /// Leak conductance (mS/cm^2).
    pub g_l: f64,
    // Reversal potentials
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
    /// Sub-step (ms). Each call runs `floor(0.5 / max(dt, 0.001))` sub-steps, about 0.5 ms;
    /// a `dt` above 0.5 ms advances nothing.
    pub dt: f64,
    /// Spike-detection threshold (mV).
    pub v_threshold: f64,
}

impl ChandelierNeuron {
    /// Create a cell at -65 mV with the default gates and parameters and `dt` = 0.01 ms.
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            d: 0.0,
            p: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_kv1: 3.0, // Kv1 delay current (slower)
            g_kv3: 4.0, // Kv3.1 for AP sharpening
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }

    /// Advance about 0.5 ms of forward-Euler sub-steps (see `dt`) with the input `current`
    /// (uA/cm^2); return 1 when `v` crosses `v_threshold` upwards between the start and the
    /// end of the call, else 0.
    ///
    /// Each sub-step updates `h`, `n`, the Kv1 gate `d` (time constant 150 ms) and the
    /// Kv3.1 gate `p` (1 ms divided by `phi`) from the present `v`. It then advances `v` with
    /// the sodium, delayed-rectifier, Kv1 (`d^4`), Kv3.1 (`p`) and leak currents. `v` is
    /// never reset. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let n_sub = (0.5 / self.dt.max(0.001)) as usize;
        for _ in 0..n_sub {
            // Wang-Buzsáki gating
            let am = safe_rate(0.1, 35.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(self.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 44.0) / 80.0).exp();

            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt;

            // Kv1 (D-type): slow activation → first-spike delay
            let d_inf = 1.0 / (1.0 + (-(self.v + 50.0) / 10.0).exp());
            let tau_d = 150.0;
            self.d += (d_inf - self.d) / tau_d * self.dt;

            // Kv3.1: fast activation
            let p_inf = 1.0 / (1.0 + (-(self.v + 10.0) / 10.0).exp());
            self.p += self.phi * (p_inf - self.p) / 1.0 * self.dt;

            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_kv1 = self.g_kv1 * self.d.powi(4) * (self.v - self.e_k);
            let i_kv3 = self.g_kv3 * self.p * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);

            self.v += (-i_na - i_k - i_kv1 - i_kv3 - i_l + current) / self.c_m * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    /// Restore the state (`v`, `h`, `n`, `d`, `p`) to its initial values; the parameters are
    /// kept.
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.8;
        self.n = 0.1;
        self.d = 0.0;
        self.p = 0.0;
    }
}

impl Default for ChandelierNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Cerebellar Basket Cell
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::PVFastSpikingNeuron;
    use super::*;
    use crate::neurons::host_budget;

    #[test]
    fn chandelier_fires_with_input() {
        let mut n = ChandelierNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(spikes > 0, "Chandelier must fire with sustained input");
    }

    #[test]
    fn chandelier_no_fire_without_input() {
        let mut n = ChandelierNeuron::new();
        let spikes: i32 = (0..2000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn chandelier_has_kv1_delay_current() {
        // Chandelier has Kv1 (D-type) which activates slowly.
        // After sustained input, Kv1 contributes extra K+ current → lower steady-state rate.
        let mut ch = ChandelierNeuron::new();
        let mut pv = PVFastSpikingNeuron::new();
        let ch_spikes: i32 = (0..5000).map(|_| ch.step(3.0)).sum();
        let pv_spikes: i32 = (0..5000).map(|_| pv.step(3.0)).sum();
        // Both should fire, Chandelier may fire fewer due to extra K+
        assert!(ch_spikes > 0, "Chandelier must fire");
        assert!(pv_spikes > 0, "PV+ must fire");
        assert!(
            ch_spikes <= pv_spikes + 10,
            "Chandelier ({ch_spikes}) should fire <= PV+ ({pv_spikes}) due to Kv1"
        );
    }

    #[test]
    fn chandelier_reset_roundtrip() {
        let mut n = ChandelierNeuron::new();
        for _ in 0..1000 {
            n.step(3.0);
        }
        n.reset();
        let mut fresh = ChandelierNeuron::new();
        let r1: i32 = (0..500).map(|_| n.step(3.0)).sum();
        let r2: i32 = (0..500).map(|_| fresh.step(3.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn chandelier_voltage_bounded() {
        let mut n = ChandelierNeuron::new();
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn chandelier_performance_5k_steps() {
        let mut n = ChandelierNeuron::new();
        host_budget::assert_within(module_path!(), "chandelier_performance_5k_steps", || {
            for _ in 0..5_000 {
                std::hint::black_box(n.step(3.0));
            }
        });
    }
}
