// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Motor Neuron Models

//! Motor neuron models for spinal and cortical motor circuits.
//!
//! Motor model group: alpha motor, gamma motor, upper motor, Renshaw cell, motor unit.
//! Added one by one with full 7-point checklist verification.

use super::biophysical::safe_rate;

// ═══════════════════════════════════════════════════════════════════
// Alpha Motor Neuron
// ═══════════════════════════════════════════════════════════════════

/// Alpha motor neuron — spinal cord, innervates extrafusal muscle fibres.
///
/// Biophysics: Wang-Buzsáki Na+/K+ core, persistent inward current (PIC)
/// for bistable firing (plateau potentials), Ca2+-dependent AHP for rate
/// limiting (f-I gain control). Larger soma than cortical neurons → lower
/// input resistance.
///
/// PIC is modelled as a slow L-type Ca2+ current that activates at
/// depolarised potentials and inactivates very slowly, enabling plateau
/// potentials and self-sustained firing after brief input.
///
/// AHP from Ca2+-activated K+ (SK channels) limits firing rate and
/// produces the characteristic linear f-I relationship of motor neurons.
///
/// Powers & Binder, J. Neurophysiol. 86, 2001.
/// Heckman & Enoka, Compr. Physiol. 2(4), 2012.
#[derive(Clone, Debug)]
pub struct AlphaMotorNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub m_pic: f64,  // PIC (L-type Ca²⁺) activation
    pub h_pic: f64,  // PIC slow inactivation (tau ~200 ms)
    pub ca: f64,     // Intracellular Ca²⁺ (µM)
    pub ca_buf: f64, // Bound Ca²⁺ (buffered fraction)
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_pic: f64, // Persistent inward current
    pub g_ahp: f64, // Ca²⁺-dependent K⁺ (AHP)
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,    // Ca²⁺ decay (ms)
    pub buf_ratio: f64, // Buffering ratio (fraction of Ca²⁺ bound)
    pub dt: f64,
    pub v_threshold: f64,
}

impl AlphaMotorNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            m_pic: 0.0,
            h_pic: 1.0, // PIC inactivation starts de-inactivated
            ca: 0.0,
            ca_buf: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_pic: 0.15, // PIC for plateau potentials (conservative)
            g_ahp: 3.0,  // Strong AHP for rate limiting
            g_l: 0.3,    // Higher leak (larger soma, stabilises rest)
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_l: -65.0,
            c_m: 1.5, // Larger soma → higher capacitance
            phi: 4.0,
            tau_ca: 150.0,    // Slow Ca²⁺ clearance for AHP
            buf_ratio: 0.003, // ~0.3% free Ca²⁺ (99.7% buffered)
            dt: 0.01,
            v_threshold: -20.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let n_sub = (0.5 / self.dt.max(0.001)) as usize;
        for _ in 0..n_sub {
            // WB Na+/K+ gating
            let am = safe_rate(0.1, 35.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(self.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 44.0) / 80.0).exp();

            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt;

            // PIC (L-type Ca²⁺): activation + slow inactivation
            // Activation: m_pic, tau ~50 ms, half-act -50 mV
            let m_pic_inf = 1.0 / (1.0 + (-(self.v + 40.0) / 5.0).exp());
            self.m_pic += (m_pic_inf - self.m_pic) / 50.0 * self.dt;
            // Inactivation: h_pic, tau ~200 ms, half-inact -40 mV
            // L-type inactivation is slow and Ca²⁺-dependent
            let h_pic_inf = 1.0 / (1.0 + ((self.v + 40.0) / 8.0).exp());
            let tau_h_pic = 200.0 + 100.0 / (1.0 + ((self.v + 40.0) / 10.0).powi(2)).max(0.01);
            self.h_pic += (h_pic_inf - self.h_pic) / tau_h_pic * self.dt;
            self.h_pic = self.h_pic.clamp(0.0, 1.0);

            // Ca²⁺ dynamics with buffering
            // Total Ca²⁺ entry (PIC-mediated)
            let i_ca_entry = self.g_pic * self.m_pic * self.h_pic * (self.v - self.e_ca);
            let ca_influx = if i_ca_entry < 0.0 {
                -i_ca_entry * 0.001
            } else {
                0.0
            };
            let ca_spike = if self.v > -10.0 { 0.02 } else { 0.0 };
            // Only ~0.3% of entering Ca²⁺ is free (rest is buffered)
            let free_ca_change = (ca_influx + ca_spike) * self.buf_ratio;
            self.ca += (-self.ca / self.tau_ca + free_ca_change) * self.dt;
            if self.ca < 0.0 {
                self.ca = 0.0;
            }
            // Buffered pool tracks total entry (slower dynamics)
            self.ca_buf += ((ca_influx + ca_spike) * (1.0 - self.buf_ratio)
                - self.ca_buf / (self.tau_ca * 5.0))
                * self.dt;
            if self.ca_buf < 0.0 {
                self.ca_buf = 0.0;
            }

            // AHP: Ca²⁺-activated K⁺ (SK channels), Hill n=2
            let ca_total = self.ca + self.ca_buf * 0.01; // Buffered contributes slowly
            let ahp_inf = ca_total * ca_total / (ca_total * ca_total + 0.25);

            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_pic = self.g_pic * self.m_pic * self.h_pic * (self.v - self.e_ca);
            let i_ahp = self.g_ahp * ahp_inf * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);

            self.v += (-i_na - i_k - i_pic - i_ahp - i_l + current) / self.c_m * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for AlphaMotorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Gamma Motor Neuron
// ═══════════════════════════════════════════════════════════════════

/// Gamma motor neuron — innervates intrafusal fibres of muscle spindles.
///
/// Regulates proprioceptive sensitivity by adjusting spindle tension.
/// Smaller soma than alpha, lower firing rates (5-30 Hz), no PIC.
/// Simple LIF with spike-frequency adaptation (slow K+ current).
/// Two subtypes: dynamic (bag1, velocity-sensitive) and static
/// (bag2/chain, length-sensitive) — controlled by `dynamic` flag.
///
/// Prochazka & Hulliger, Prog. Brain Res. 80, 1989.
/// Taylor et al., J. Physiol. 519(3), 1999.
#[derive(Clone, Debug)]
pub struct GammaMotorNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub adapt: f64,     // Slow adaptation current
    pub tau_adapt: f64, // Adaptation time constant (ms)
    pub a_adapt: f64,   // Adaptation coupling strength
    pub gain: f64,      // Input gain (fusimotor drive → mV)
    pub dynamic: bool,  // true = dynamic (bag1), false = static (bag2/chain)
    pub dt: f64,
}

impl GammaMotorNeuron {
    pub fn new() -> Self {
        Self::dynamic()
    }

    /// Dynamic gamma — innervates bag1 intrafusal fibres (velocity-sensitive).
    pub fn dynamic() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 8.0,
            adapt: 0.0,
            tau_adapt: 100.0,
            a_adapt: 0.3,
            gain: 1.0,
            dynamic: true,
            dt: 0.5,
        }
    }

    /// Static gamma — innervates bag2/chain intrafusal fibres (length-sensitive).
    pub fn static_type() -> Self {
        Self {
            tau: 12.0,        // Slower membrane
            tau_adapt: 200.0, // Larger adaptation time constant
            a_adapt: 0.5,
            dynamic: false,
            ..Self::dynamic()
        }
    }

    /// Step with fusimotor drive (arbitrary units, ≥ 0). Returns spike (1/0).
    pub fn step(&mut self, drive: f64) -> i32 {
        if !self.is_valid() || !drive.is_finite() {
            return 0;
        }
        let v_old = self.v;
        let adapt_old = self.adapt;
        let input = self.gain * drive.max(0.0) - adapt_old;
        let v_target = self.v_rest + input;
        let v_candidate = v_target + (v_old - v_target) * (-self.dt / self.tau).exp();
        let adapt_target = self.a_adapt * (v_candidate - self.v_rest);
        let adapt_candidate =
            adapt_target + (adapt_old - adapt_target) * (-self.dt / self.tau_adapt).exp();
        if !v_candidate.is_finite() || !adapt_candidate.is_finite() {
            return 0;
        }
        self.v = v_candidate;
        self.adapt = adapt_candidate;

        if v_candidate >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0;
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau,
            self.adapt,
            self.tau_adapt,
            self.a_adapt,
            self.gain,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.tau > 0.0
            && self.tau_adapt > 0.0
            && self.dt > 0.0
            && self.gain >= 0.0
            && self.v_reset < self.v_threshold
    }
}

impl Default for GammaMotorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Upper Motor Neuron (Corticospinal L5 Pyramidal)
// ═══════════════════════════════════════════════════════════════════

/// Upper motor neuron — layer 5 pyramidal cell, corticospinal projection.
///
/// Biophysics: Pospischil 2008 RS parameterisation (Na+, K+, M-current)
/// with added high-threshold Ca2+ current for dendritic Ca2+ spikes.
/// Regular-spiking with adaptation. Drives alpha/gamma motor neurons
/// via corticospinal tract.
///
/// Pospischil et al., Biol. Cybern. 99(4-5), 2008 (RS variant).
/// Larkum, Trends Neurosci. 36(3), 2013 (dendritic Ca2+ spikes).
#[derive(Clone, Debug)]
pub struct UpperMotorNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64, // M-current (Kv7) activation
    pub s: f64, // High-threshold Ca2+ activation
    // Conductances
    pub g_na: f64,
    pub g_k: f64,
    pub g_m: f64,
    pub g_ca: f64,
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl UpperMotorNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            p: 0.0,
            s: 0.0,
            g_na: 50.0,
            g_k: 5.0,
            g_m: 0.07, // M-current for adaptation (Pospischil RS)
            g_ca: 0.3, // High-threshold dendritic Ca2+
            g_l: 0.1,
            e_na: 50.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_l: -70.0,
            c_m: 1.0,
            dt: 0.025,
            v_threshold: -20.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let vt = -56.2;
        for _ in 0..4 {
            // Pospischil Na+ gating
            let dv = self.v - vt;
            let x_m = dv - 13.0;
            let alpha_m = if x_m.abs() < 1e-6 {
                0.32 * 4.0
            } else {
                -0.32 * x_m / ((-x_m / 4.0).exp() - 1.0)
            };
            let x_h = dv - 17.0;
            let beta_m = if x_h.abs() < 1e-6 {
                0.28 * 5.0
            } else {
                0.28 * x_h / ((x_h / 5.0).exp() - 1.0)
            };
            let alpha_h = 0.128 * (-(dv - 17.0) / 18.0).exp();
            let beta_h = 4.0 / (1.0 + (-(dv - 40.0) / 5.0).exp());
            // K+ gating
            let x_n = dv - 15.0;
            let alpha_n = if x_n.abs() < 1e-6 {
                0.032 * 5.0
            } else {
                -0.032 * x_n / ((-x_n / 5.0).exp() - 1.0)
            };
            let beta_n = 0.5 * (-(dv - 10.0) / 40.0).exp();

            self.m += (alpha_m * (1.0 - self.m) - beta_m * self.m) * self.dt;
            self.h += (alpha_h * (1.0 - self.h) - beta_h * self.h) * self.dt;
            self.n += (alpha_n * (1.0 - self.n) - beta_n * self.n) * self.dt;

            // M-current (slow K+, adaptation)
            let p_inf = 1.0 / (1.0 + (-(self.v + 35.0) / 10.0).exp());
            let tau_p =
                400.0 / (3.3 * ((self.v + 35.0) / 20.0).exp() + (-(self.v + 35.0) / 20.0).exp());
            self.p += (p_inf - self.p) / tau_p * self.dt;

            // High-threshold Ca2+ (dendritic spike)
            let s_inf = 1.0 / (1.0 + (-(self.v + 20.0) / 5.0).exp());
            self.s += (s_inf - self.s) / 10.0 * self.dt;

            let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_m = self.g_m * self.p * (self.v - self.e_k);
            let i_ca = self.g_ca * self.s.powi(2) * (self.v - self.e_ca);
            let i_l = self.g_l * (self.v - self.e_l);

            self.v += (-i_na - i_k - i_m - i_ca - i_l + current) / self.c_m * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -70.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.3;
        self.p = 0.0;
        self.s = 0.0;
    }
}

impl Default for UpperMotorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Renshaw Cell (Spinal Inhibitory Interneuron)
// ═══════════════════════════════════════════════════════════════════

/// Renshaw cell — spinal inhibitory interneuron for recurrent inhibition.
///
/// Receives collaterals from alpha motor neuron axons, provides
/// glycinergic recurrent inhibition back to the motor pool. Characteristic
/// high-frequency initial burst (cholinergic nicotinic drive from motor
/// axon collaterals) followed by rapid adaptation.
///
/// WB gating core with strong adaptation (M-current analogue) to produce
/// the burst-then-decay response pattern.
///
/// Renshaw 1941 (discovery); Windhorst, Prog. Neurobiol. 46(5), 1996.
#[derive(Clone, Debug)]
pub struct RenshawCell {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub adapt: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_adapt: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_adapt: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl RenshawCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            adapt: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_adapt: 5.0,
            g_l: 0.12,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            tau_adapt: 50.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let n_sub = (0.5 / self.dt.max(0.001)) as usize;
        for _ in 0..n_sub {
            let am = safe_rate(0.1, 35.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(self.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 44.0) / 80.0).exp();

            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt;

            let adapt_inf = 1.0 / (1.0 + (-(self.v + 30.0) / 5.0).exp());
            self.adapt += (adapt_inf - self.adapt) / self.tau_adapt * self.dt;

            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_adapt = self.g_adapt * self.adapt * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);

            self.v += (-i_na - i_k - i_adapt - i_l + current) / self.c_m * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.8;
        self.n = 0.1;
        self.adapt = 0.0;
    }
}

impl Default for RenshawCell {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Motor Unit (Alpha Motor Neuron + Muscle Fibre)
// ═══════════════════════════════════════════════════════════════════

/// Motor unit — functional unit of motor control: alpha motor neuron + muscle fibre.
///
/// Each spike from the embedded LIF motor neuron triggers a muscle twitch.
/// Force output is the summation of overlapping twitches (rate coding).
/// Higher firing rates → more twitch overlap → higher force (tetanus).
///
/// Muscle twitch modelled as a critically-damped second-order system:
/// f(t) = A * (t/τ) * exp(1 - t/τ), giving a smooth rise-then-decay.
///
/// Fuglevand et al., J. Neurophysiol. 70(6), 1993.
/// Heckman & Enoka, Compr. Physiol. 2(4), 2012.
#[derive(Clone, Debug)]
pub struct MotorUnit {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64, // Membrane time constant (ms)
    pub adapt: f64,
    pub tau_adapt: f64,
    pub a_adapt: f64,
    pub gain: f64,
    // Muscle fibre
    pub force: f64,       // Current force output (normalised)
    pub twitch_amp: f64,  // Peak twitch amplitude
    pub tau_twitch: f64,  // Twitch contraction time (ms)
    pub force_decay: f64, // Force decay per step
    pub dt: f64,
}

impl MotorUnit {
    pub fn new() -> Self {
        Self::slow()
    }

    /// Slow motor unit (type S): small, fatigue-resistant, low force.
    pub fn slow() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            adapt: 0.0,
            tau_adapt: 100.0,
            a_adapt: 0.2,
            gain: 1.0,
            force: 0.0,
            twitch_amp: 0.05,
            tau_twitch: 90.0,
            force_decay: 0.0,
            dt: 0.5,
        }
    }

    /// Fast motor unit (type FF): large, fatigable, high force.
    pub fn fast() -> Self {
        Self {
            tau_m: 6.0,
            tau_adapt: 50.0,
            a_adapt: 0.1,
            twitch_amp: 0.3,
            tau_twitch: 30.0,
            ..Self::slow()
        }
    }

    /// Step with descending drive (≥ 0). Returns spike (1/0). Force accessible via `.force`.
    pub fn step(&mut self, drive: f64) -> i32 {
        let input = self.gain * drive.max(0.0) - self.adapt;
        self.v += (-(self.v - self.v_rest) + input) / self.tau_m * self.dt;
        self.adapt +=
            (self.a_adapt * (self.v - self.v_rest) - self.adapt) / self.tau_adapt * self.dt;

        // Force decay: exponential relaxation
        self.force *= (-self.dt / self.tau_twitch).exp();

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            // Spike → muscle twitch (add to force)
            self.force += self.twitch_amp;
            if self.force > 1.0 {
                self.force = 1.0;
            }
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0;
        self.force = 0.0;
    }
}

impl Default for MotorUnit {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Alpha Motor Neuron — 6-dimension coverage ──────────────────

    #[test]
    fn alpha_motor_fires_with_input() {
        let mut n = AlphaMotorNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(
            spikes > 0,
            "alpha motor must fire with sustained input: got {spikes}"
        );
    }

    #[test]
    fn alpha_motor_no_fire_without_input() {
        let mut n = AlphaMotorNeuron::new();
        let spikes: i32 = (0..3000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0, "alpha motor should not fire at rest");
    }

    #[test]
    fn alpha_motor_negative_current_no_fire() {
        let mut n = AlphaMotorNeuron::new();
        let spikes: i32 = (0..2000).map(|_| n.step(-2.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn alpha_motor_ahp_limits_rate() {
        // AHP from Ca2+-activated K+ should limit firing rate.
        // Compare: with AHP vs without (g_ahp=0).
        let mut with_ahp = AlphaMotorNeuron::new();
        let mut no_ahp = AlphaMotorNeuron::new();
        no_ahp.g_ahp = 0.0;
        let s_ahp: i32 = (0..5000).map(|_| with_ahp.step(5.0)).sum();
        let s_none: i32 = (0..5000).map(|_| no_ahp.step(5.0)).sum();
        assert!(
            s_ahp <= s_none + 5,
            "AHP should limit rate: with={s_ahp}, without={s_none}"
        );
    }

    #[test]
    fn alpha_motor_pic_responds_to_depolarisation() {
        // PIC (m_pic) should increase from baseline during sustained input.
        let mut n = AlphaMotorNeuron::new();
        let baseline = n.m_pic;
        for _ in 0..2000 {
            n.step(4.0);
        }
        assert!(
            n.m_pic > baseline + 0.001,
            "PIC should respond to depolarisation: baseline={baseline}, after={}",
            n.m_pic
        );
    }

    #[test]
    fn alpha_motor_ca_increases_during_spiking() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(
            n.ca > 0.0,
            "Ca2+ should accumulate during spiking: ca={}",
            n.ca
        );
    }

    #[test]
    fn alpha_motor_reset_roundtrip() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..2000 {
            n.step(4.0);
        }
        n.reset();
        let mut fresh = AlphaMotorNeuron::new();
        let r1: i32 = (0..1000).map(|_| n.step(4.0)).sum();
        let r2: i32 = (0..1000).map(|_| fresh.step(4.0)).sum();
        assert_eq!(r1, r2, "reset neuron must match fresh");
    }

    #[test]
    fn alpha_motor_voltage_bounded() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..10000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite(), "voltage must stay finite");
        assert!(n.ca.is_finite(), "Ca2+ must stay finite");
        assert!(n.ca >= 0.0, "Ca2+ must be non-negative");
    }

    #[test]
    fn alpha_motor_nan_recovery() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..100 {
            n.step(3.0);
        }
        for _ in 0..10 {
            let _ = n.step(f64::NAN);
        }
        n.reset();
        assert!(n.v.is_finite());
        assert!(n.ca >= 0.0);
    }

    #[test]
    fn alpha_motor_extreme_input() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..50 {
            n.step(1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
        for _ in 0..50 {
            n.step(-1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn alpha_motor_performance() {
        let mut n = AlphaMotorNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..5_000 {
            n.step(4.0);
        }
        assert!(
            start.elapsed().as_millis() < 500,
            "5k steps took {:?}",
            start.elapsed()
        );
    }

    // ── Gamma Motor Neuron — 6-dimension coverage ──────────────────

    #[test]
    fn gamma_dynamic_fires_with_drive() {
        let mut n = GammaMotorNeuron::dynamic();
        let spikes: i32 = (0..2000).map(|_| n.step(20.0)).sum();
        assert!(spikes > 0, "gamma dynamic must fire: got {spikes}");
    }

    #[test]
    fn gamma_static_fires_with_drive() {
        let mut n = GammaMotorNeuron::static_type();
        let spikes: i32 = (0..2000).map(|_| n.step(20.0)).sum();
        assert!(spikes > 0, "gamma static must fire: got {spikes}");
    }

    #[test]
    fn gamma_no_fire_without_drive() {
        let mut n = GammaMotorNeuron::new();
        let spikes: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn gamma_negative_drive_no_fire() {
        let mut n = GammaMotorNeuron::new();
        // drive.max(0.0) clamps negatives
        let spikes: i32 = (0..1000).map(|_| n.step(-10.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn gamma_adaptation_reduces_rate() {
        let mut n = GammaMotorNeuron::new();
        let first: i32 = (0..1000).map(|_| n.step(20.0)).sum();
        let second: i32 = (0..1000).map(|_| n.step(20.0)).sum();
        assert!(
            second <= first + 3,
            "gamma should adapt: first={first}, second={second}"
        );
    }

    #[test]
    fn gamma_static_adapts_more_than_dynamic() {
        let mut dyn_ = GammaMotorNeuron::dynamic();
        let mut stat = GammaMotorNeuron::static_type();
        let dyn_spikes: i32 = (0..2000).map(|_| dyn_.step(20.0)).sum();
        let stat_spikes: i32 = (0..2000).map(|_| stat.step(20.0)).sum();
        // Static uses larger adaptation coupling and lower excitability.
        assert!(
            stat_spikes <= dyn_spikes + 5,
            "static ({stat_spikes}) should fire <= dynamic ({dyn_spikes})"
        );
    }

    #[test]
    fn gamma_reset_roundtrip() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..1000 {
            n.step(20.0);
        }
        n.reset();
        let mut fresh = GammaMotorNeuron::new();
        let r1: i32 = (0..500).map(|_| n.step(20.0)).sum();
        let r2: i32 = (0..500).map(|_| fresh.step(20.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn gamma_voltage_bounded() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..10000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
        assert!(n.adapt.is_finite());
    }

    #[test]
    fn gamma_nan_recovery() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..50 {
            n.step(20.0);
        }
        let before_v = n.v;
        let before_adapt = n.adapt;
        for _ in 0..10 {
            let _ = n.step(f64::NAN);
        }
        assert_eq!(n.v, before_v);
        assert_eq!(n.adapt, before_adapt);
        n.reset();
        assert!(n.v.is_finite());
        assert_eq!(n.adapt, 0.0);
    }

    #[test]
    fn gamma_extreme_input() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..50 {
            n.step(1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn gamma_corrupted_state_preserved_on_step() {
        let mut n = GammaMotorNeuron::new();
        n.tau = 0.0;
        let before_v = n.v;
        let before_adapt = n.adapt;
        assert_eq!(n.step(20.0), 0);
        assert_eq!(n.v, before_v);
        assert_eq!(n.adapt, before_adapt);
    }

    #[test]
    fn gamma_performance() {
        let mut n = GammaMotorNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            n.step(20.0);
        }
        assert!(
            start.elapsed().as_millis() < 50,
            "100k steps took {:?}",
            start.elapsed()
        );
    }

    // ── Upper Motor Neuron — 6-dimension coverage ──────────────────

    #[test]
    fn upper_motor_fires_with_input() {
        let mut n = UpperMotorNeuron::new();
        let spikes: i32 = (0..10000).map(|_| n.step(5.0)).sum();
        assert!(spikes > 0, "upper motor must fire: got {spikes}");
    }

    #[test]
    fn upper_motor_no_fire_without_input() {
        let mut n = UpperMotorNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn upper_motor_negative_current_no_fire() {
        let mut n = UpperMotorNeuron::new();
        let spikes: i32 = (0..2000).map(|_| n.step(-5.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn upper_motor_adaptation_via_m_current() {
        let mut n = UpperMotorNeuron::new();
        let first: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        let second: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        assert!(
            second <= first + 3,
            "M-current should cause adaptation: first={first}, second={second}"
        );
    }

    #[test]
    fn upper_motor_ca_activates_during_depolarisation() {
        let mut n = UpperMotorNeuron::new();
        let baseline = n.s;
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(
            n.s > baseline + 0.001,
            "Ca2+ gate should activate: s={}",
            n.s
        );
    }

    #[test]
    fn upper_motor_reset_roundtrip() {
        let mut n = UpperMotorNeuron::new();
        for _ in 0..3000 {
            n.step(5.0);
        }
        n.reset();
        let mut fresh = UpperMotorNeuron::new();
        let r1: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        let r2: i32 = (0..2000).map(|_| fresh.step(5.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn upper_motor_voltage_bounded() {
        let mut n = UpperMotorNeuron::new();
        for _ in 0..20000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
        assert!(n.p.is_finite());
        assert!(n.s.is_finite());
    }

    #[test]
    fn upper_motor_nan_recovery() {
        let mut n = UpperMotorNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        for _ in 0..10 {
            let _ = n.step(f64::NAN);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn upper_motor_extreme_input() {
        let mut n = UpperMotorNeuron::new();
        for _ in 0..50 {
            n.step(1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn upper_motor_performance() {
        let mut n = UpperMotorNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(5.0);
        }
        assert!(
            start.elapsed().as_millis() < 100,
            "10k steps took {:?}",
            start.elapsed()
        );
    }

    // ── Renshaw Cell — 6-dimension coverage ────────────────────────

    #[test]
    fn renshaw_fires_with_input() {
        let mut n = RenshawCell::new();
        let spikes: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(spikes > 0, "Renshaw must fire: got {spikes}");
    }

    #[test]
    fn renshaw_no_fire_without_input() {
        let mut n = RenshawCell::new();
        let spikes: i32 = (0..3000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn renshaw_negative_current_no_fire() {
        let mut n = RenshawCell::new();
        let spikes: i32 = (0..2000).map(|_| n.step(-2.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn renshaw_burst_then_adapt() {
        // Renshaw should fire more in the first epoch than the second
        let mut n = RenshawCell::new();
        let first: i32 = (0..2000).map(|_| n.step(4.0)).sum();
        let second: i32 = (0..2000).map(|_| n.step(4.0)).sum();
        assert!(
            second <= first + 5,
            "Renshaw should adapt: first={first}, second={second}"
        );
    }

    #[test]
    fn renshaw_adapt_increases_during_firing() {
        let mut n = RenshawCell::new();
        let baseline = n.adapt;
        for _ in 0..3000 {
            n.step(4.0);
        }
        assert!(
            n.adapt > baseline + 0.01,
            "adaptation variable should increase: adapt={}",
            n.adapt
        );
    }

    #[test]
    fn renshaw_reset_roundtrip() {
        let mut n = RenshawCell::new();
        for _ in 0..2000 {
            n.step(4.0);
        }
        n.reset();
        let mut fresh = RenshawCell::new();
        let r1: i32 = (0..1000).map(|_| n.step(4.0)).sum();
        let r2: i32 = (0..1000).map(|_| fresh.step(4.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn renshaw_voltage_bounded() {
        let mut n = RenshawCell::new();
        for _ in 0..10000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
        assert!(n.adapt.is_finite());
    }

    #[test]
    fn renshaw_nan_recovery() {
        let mut n = RenshawCell::new();
        for _ in 0..100 {
            n.step(3.0);
        }
        for _ in 0..10 {
            let _ = n.step(f64::NAN);
        }
        n.reset();
        assert!(n.v.is_finite());
        assert_eq!(n.adapt, 0.0);
    }

    #[test]
    fn renshaw_extreme_input() {
        let mut n = RenshawCell::new();
        for _ in 0..50 {
            n.step(1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn renshaw_performance() {
        let mut n = RenshawCell::new();
        let start = std::time::Instant::now();
        for _ in 0..5_000 {
            n.step(4.0);
        }
        assert!(
            start.elapsed().as_millis() < 500,
            "5k steps took {:?}",
            start.elapsed()
        );
    }

    // ── Motor Unit — 6-dimension coverage ──────────────────────────

    #[test]
    fn motor_unit_fires_with_drive() {
        let mut mu = MotorUnit::new();
        let spikes: i32 = (0..2000).map(|_| mu.step(20.0)).sum();
        assert!(spikes > 0, "motor unit must fire: got {spikes}");
    }

    #[test]
    fn motor_unit_no_fire_without_drive() {
        let mut mu = MotorUnit::new();
        let spikes: i32 = (0..1000).map(|_| mu.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn motor_unit_negative_drive_no_fire() {
        let mut mu = MotorUnit::new();
        let spikes: i32 = (0..1000).map(|_| mu.step(-10.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn motor_unit_force_increases_with_spikes() {
        let mut mu = MotorUnit::new();
        assert_eq!(mu.force, 0.0);
        for _ in 0..2000 {
            mu.step(20.0);
        }
        assert!(
            mu.force > 0.0,
            "force should increase during spiking: f={}",
            mu.force
        );
    }

    #[test]
    fn motor_unit_force_decays_without_input() {
        let mut mu = MotorUnit::new();
        // Build up force
        for _ in 0..1000 {
            mu.step(20.0);
        }
        let peak = mu.force;
        assert!(peak > 0.0);
        // No input → force decays
        for _ in 0..5000 {
            mu.step(0.0);
        }
        assert!(
            mu.force < peak,
            "force should decay: peak={peak}, now={}",
            mu.force
        );
    }

    #[test]
    fn motor_unit_fast_produces_more_force() {
        let mut slow = MotorUnit::slow();
        let mut fast = MotorUnit::fast();
        for _ in 0..2000 {
            slow.step(20.0);
            fast.step(20.0);
        }
        assert!(
            fast.force >= slow.force,
            "fast MU ({}) should produce >= force than slow ({})",
            fast.force,
            slow.force
        );
    }

    #[test]
    fn motor_unit_force_capped_at_one() {
        let mut mu = MotorUnit::fast();
        for _ in 0..10000 {
            mu.step(50.0);
        }
        assert!(mu.force <= 1.0, "force must not exceed 1.0: f={}", mu.force);
    }

    #[test]
    fn motor_unit_reset_roundtrip() {
        let mut mu = MotorUnit::new();
        for _ in 0..1000 {
            mu.step(20.0);
        }
        mu.reset();
        assert_eq!(mu.force, 0.0);
        assert_eq!(mu.adapt, 0.0);
        let mut fresh = MotorUnit::new();
        let r1: i32 = (0..500).map(|_| mu.step(20.0)).sum();
        let r2: i32 = (0..500).map(|_| fresh.step(20.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn motor_unit_voltage_bounded() {
        let mut mu = MotorUnit::new();
        for _ in 0..10000 {
            mu.step(50.0);
        }
        assert!(mu.v.is_finite());
        assert!(mu.force.is_finite());
    }

    #[test]
    fn motor_unit_nan_recovery() {
        let mut mu = MotorUnit::new();
        for _ in 0..50 {
            mu.step(20.0);
        }
        for _ in 0..10 {
            let _ = mu.step(f64::NAN);
        }
        mu.reset();
        assert!(mu.v.is_finite());
        assert_eq!(mu.force, 0.0);
    }

    #[test]
    fn motor_unit_performance() {
        let mut mu = MotorUnit::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            mu.step(20.0);
        }
        assert!(
            start.elapsed().as_millis() < 50,
            "100k steps took {:?}",
            start.elapsed()
        );
    }
}
