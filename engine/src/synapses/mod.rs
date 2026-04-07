// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic STDP Synapse

//! # Stochastic STDP Synapse
//!
//! Implements Spike-Timing-Dependent Plasticity (STDP) for stochastic bitstreams.
//! Optimized for on-chip adaptation.

use crate::neuron::mask;

/// Parameters for the STDP rule.
#[derive(Clone, Copy, Debug)]
pub struct StdpParams {
    pub a_plus: i16,
    pub a_minus: i16,
    pub decay: i16,
    pub w_min: i16,
    pub w_max: i16,
}

/// A synapse with STDP learning capability.
#[derive(Clone, Debug)]
pub struct StdpSynapse {
    /// Current weight value (fixed-point).
    pub weight: i16,
    /// Pre-synaptic trace.
    pub trace_pre: i16,
    /// Post-synaptic trace.
    pub trace_post: i16,
    /// Data width.
    pub data_width: u32,
    /// Fraction bits.
    pub fraction: u32,
}

impl StdpSynapse {
    pub fn new(initial_weight: i16, data_width: u32, fraction: u32) -> Self {
        Self {
            weight: initial_weight,
            trace_pre: 0,
            trace_post: 0,
            data_width,
            fraction,
        }
    }

    /// Update traces and weight based on pre and post spikes.
    ///
    /// This is a simplified, hardware-friendly STDP rule:
    ///   - On pre-spike: trace_pre += A_plus; weight += trace_post * rate
    ///   - On post-spike: trace_post += A_minus; weight += trace_pre * rate
    ///   - Traces decay over time.
    pub fn step(&mut self, pre_spike: bool, post_spike: bool, params: &StdpParams) {
        // 1. Decay traces
        self.trace_pre = mask(
            (self.trace_pre as i32 * params.decay as i32) >> self.fraction,
            self.data_width,
        );
        self.trace_post = mask(
            (self.trace_post as i32 * params.decay as i32) >> self.fraction,
            self.data_width,
        );

        // 2. Update traces
        if pre_spike {
            self.trace_pre = mask(
                self.trace_pre as i32 + params.a_plus as i32,
                self.data_width,
            );
        }
        if post_spike {
            self.trace_post = mask(
                self.trace_post as i32 + params.a_minus as i32,
                self.data_width,
            );
        }

        // 3. Update weight — mutually exclusive per timestep.
        // Simultaneous spikes → LTP (pre-before-post convention).
        if post_spike {
            let dw = (self.trace_pre as i32 * params.a_plus.abs() as i32) >> self.fraction;
            let new_w = (self.weight as i32 + dw).min(params.w_max as i32);
            self.weight = mask(new_w, self.data_width);
        } else if pre_spike {
            let dw = (self.trace_post as i32 * params.a_minus.abs() as i32) >> self.fraction;
            let new_w = (self.weight as i32 - dw).max(params.w_min as i32);
            self.weight = mask(new_w, self.data_width);
        }
    }
}

/// Reward-modulated STDP synapse.
///
/// Eligibility trace accumulates Hebbian coincidences; weight update
/// fires only when a global reward signal arrives.
/// Izhikevich, Cerebral Cortex 17(10), 2007.
#[derive(Clone, Debug)]
pub struct RewardStdpSynapse {
    pub weight: f64,
    pub w_min: f64,
    pub w_max: f64,
    pub eligibility: f64,
    pub trace_decay: f64,
    pub anti_hebbian_scale: f64,
    pub learning_rate: f64,
}

impl RewardStdpSynapse {
    pub fn new(w: f64, w_min: f64, w_max: f64) -> Self {
        Self {
            weight: w,
            w_min,
            w_max,
            eligibility: 0.0,
            trace_decay: 0.95,
            anti_hebbian_scale: 0.5,
            learning_rate: 0.01,
        }
    }

    /// Accumulate eligibility trace from pre/post spike coincidence.
    pub fn step(&mut self, pre: bool, post: bool) {
        if pre && post {
            self.eligibility += 1.0;
        } else if pre && !post {
            self.eligibility -= self.anti_hebbian_scale;
        }
        self.eligibility *= self.trace_decay;
    }

    /// Apply reward signal: weight += lr * reward * eligibility.
    pub fn apply_reward(&mut self, reward: f64) {
        let update = self.learning_rate * reward * self.eligibility;
        self.weight = (self.weight + update).clamp(self.w_min, self.w_max);
    }
}

/// Static synapse with excitatory/inhibitory polarity.
#[derive(Clone, Debug)]
pub struct StaticSynapse {
    pub weight: f64,
    pub is_excitatory: bool,
    pub delay: u32,
}

impl StaticSynapse {
    pub fn new(weight: f64, is_excitatory: bool) -> Self {
        Self {
            weight: weight.abs(),
            is_excitatory,
            delay: 0,
        }
    }

    /// Compute post-synaptic current from pre-synaptic spike.
    pub fn transmit(&self, pre_spike: bool) -> f64 {
        if !pre_spike {
            return 0.0;
        }
        if self.is_excitatory {
            self.weight
        } else {
            -self.weight
        }
    }
}

/// Triplet STDP: considers timing of three spikes for more accurate
/// visual cortex plasticity modelling.
///
/// Extends pair-based STDP with a slow post-synaptic trace o2 that gates
/// potentiation and a slow pre-synaptic trace r2 that gates depression:
///
///   Δw⁺ = A₂⁺ · r1(t) · o2(t-ε)   (triplet LTP)
///   Δw⁻ = A₂⁻ · o1(t) · r2(t-ε)   (triplet LTD)
///
/// Reference: Pfister & Gerstner (2006) "Triplet of spikes in a model of
/// spike timing-dependent plasticity", J. Neuroscience 26(38).
#[derive(Clone, Debug)]
pub struct TripletStdpSynapse {
    pub weight: f64,
    pub w_min: f64,
    pub w_max: f64,
    // Fast traces (pair-based).
    pub r1: f64,
    pub o1: f64,
    // Slow traces (triplet).
    pub r2: f64,
    pub o2: f64,
    // Time constants.
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub tau_x: f64,
    pub tau_y: f64,
    // Amplitudes.
    pub a2_plus: f64,
    pub a2_minus: f64,
    pub a3_plus: f64,
    pub a3_minus: f64,
    pub dt: f64,
}

impl TripletStdpSynapse {
    pub fn new(weight: f64, w_min: f64, w_max: f64) -> Self {
        Self {
            weight,
            w_min,
            w_max,
            r1: 0.0,
            o1: 0.0,
            r2: 0.0,
            o2: 0.0,
            tau_plus: 16.8,
            tau_minus: 33.7,
            tau_x: 101.0,
            tau_y: 125.0,
            a2_plus: 0.005,
            a2_minus: 0.007,
            a3_plus: 0.006,
            a3_minus: 0.002,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, pre_spike: bool, post_spike: bool) {
        // Decay all traces.
        self.r1 *= (-self.dt / self.tau_plus).exp();
        self.o1 *= (-self.dt / self.tau_minus).exp();
        self.r2 *= (-self.dt / self.tau_x).exp();
        self.o2 *= (-self.dt / self.tau_y).exp();

        if pre_spike {
            // LTD: depression gated by slow pre-trace.
            let dw_minus = -(self.a2_minus + self.a3_minus * self.r2) * self.o1;
            self.weight = (self.weight + dw_minus).clamp(self.w_min, self.w_max);
            self.r1 += 1.0;
            self.r2 += 1.0;
        }

        if post_spike {
            // LTP: potentiation gated by slow post-trace.
            let dw_plus = (self.a2_plus + self.a3_plus * self.o2) * self.r1;
            self.weight = (self.weight + dw_plus).clamp(self.w_min, self.w_max);
            self.o1 += 1.0;
            self.o2 += 1.0;
        }
    }
}

/// Short-term plasticity (STP): facilitation and depression.
///
/// Tsodyks-Markram model of use-dependent synaptic dynamics on ms-to-s timescale.
///
///   dx/dt = (1 - x) / τ_d - u · x · δ(t_spike)
///   du/dt = (U - u) / τ_f + U · (1 - u) · δ(t_spike)
///   PSC = A · u · x · δ(t_spike)
///
/// x: available resources (depression), u: release probability (facilitation).
///
/// Reference: Tsodyks & Markram (1997), Markram et al. (1998).
#[derive(Clone, Debug)]
pub struct ShortTermPlasticitySynapse {
    pub x: f64,
    pub u: f64,
    pub u_base: f64,
    pub tau_d: f64,
    pub tau_f: f64,
    pub amplitude: f64,
    pub dt: f64,
}

impl ShortTermPlasticitySynapse {
    /// Depressing synapse (default).
    pub fn new_depressing() -> Self {
        Self {
            x: 1.0,
            u: 0.5,
            u_base: 0.5,
            tau_d: 200.0,
            tau_f: 20.0,
            amplitude: 1.0,
            dt: 1.0,
        }
    }

    /// Facilitating synapse.
    pub fn new_facilitating() -> Self {
        Self {
            x: 1.0,
            u: 0.1,
            u_base: 0.1,
            tau_d: 50.0,
            tau_f: 500.0,
            amplitude: 1.0,
            dt: 1.0,
        }
    }

    /// Step: returns post-synaptic current.
    pub fn step(&mut self, pre_spike: bool) -> f64 {
        // Recover between spikes.
        self.x += (1.0 - self.x) / self.tau_d * self.dt;
        self.u += (self.u_base - self.u) / self.tau_f * self.dt;

        if pre_spike {
            // Facilitation: increase release probability.
            self.u += self.u_base * (1.0 - self.u);
            // Compute PSC before depression.
            let psc = self.amplitude * self.u * self.x;
            // Depression: consume resources.
            self.x -= self.u * self.x;
            self.x = self.x.max(0.0);
            psc
        } else {
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.x = 1.0;
        self.u = self.u_base;
    }
}

/// Dopamine-gated STDP: learning rate modulated by global reward signal.
///
/// Standard STDP with an eligibility trace gated by dopamine concentration.
/// Weight updates only occur when dopamine is present:
///
///   dw/dt = DA(t) · e(t)
///   de/dt = -e/τ_e + STDP(Δt) · δ(t_spike)
///   dDA/dt = -DA/τ_DA + reward(t)
///
/// Reference: Izhikevich (2007) "Solving the distal reward problem through
/// linkage of STDP and dopamine signaling", Cerebral Cortex 17(10).
#[derive(Clone, Debug)]
pub struct DopamineStdpSynapse {
    pub weight: f64,
    pub w_min: f64,
    pub w_max: f64,
    pub eligibility: f64,
    pub dopamine: f64,
    pub trace_pre: f64,
    pub trace_post: f64,
    pub tau_e: f64,
    pub tau_da: f64,
    pub tau_pre: f64,
    pub tau_post: f64,
    pub a_plus: f64,
    pub a_minus: f64,
    pub lr: f64,
    pub dt: f64,
}

impl DopamineStdpSynapse {
    pub fn new(weight: f64, w_min: f64, w_max: f64) -> Self {
        Self {
            weight,
            w_min,
            w_max,
            eligibility: 0.0,
            dopamine: 0.0,
            trace_pre: 0.0,
            trace_post: 0.0,
            tau_e: 1000.0,
            tau_da: 200.0,
            tau_pre: 20.0,
            tau_post: 20.0,
            a_plus: 1.0,
            a_minus: -1.0,
            lr: 0.001,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f64) {
        // Decay traces.
        self.trace_pre *= (-self.dt / self.tau_pre).exp();
        self.trace_post *= (-self.dt / self.tau_post).exp();
        self.eligibility *= (-self.dt / self.tau_e).exp();
        self.dopamine += (-self.dopamine / self.tau_da + reward) * self.dt;

        if pre_spike {
            // LTD from accumulated post-trace.
            self.eligibility += self.a_minus * self.trace_post;
            self.trace_pre += 1.0;
        }
        if post_spike {
            // LTP from accumulated pre-trace.
            self.eligibility += self.a_plus * self.trace_pre;
            self.trace_post += 1.0;
        }

        // Dopamine-gated weight update.
        let dw = self.lr * self.dopamine * self.eligibility * self.dt;
        self.weight = (self.weight + dw).clamp(self.w_min, self.w_max);
    }

    pub fn reset(&mut self) {
        self.eligibility = 0.0;
        self.dopamine = 0.0;
        self.trace_pre = 0.0;
        self.trace_post = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> StdpParams {
        StdpParams {
            a_plus: 64,  // 0.25 in Q8.8
            a_minus: 48, // 0.1875 in Q8.8
            decay: 230,  // ~0.90 in Q8.8
            w_min: 0,
            w_max: 255,
        }
    }

    #[test]
    fn potentiation_increases_weight() {
        let mut syn = StdpSynapse::new(128, 16, 8);
        let params = default_params();
        // Multiple pre spikes to build trace
        for _ in 0..5 {
            syn.step(true, false, &params);
        }
        let w_before = syn.weight;
        // Post spike triggers LTP via accumulated pre-trace
        syn.step(false, true, &params);
        assert!(syn.weight > w_before, "LTP must increase weight");
    }

    #[test]
    fn depression_decreases_weight() {
        let mut syn = StdpSynapse::new(128, 16, 8);
        let params = default_params();
        // Multiple post spikes to build trace
        for _ in 0..5 {
            syn.step(false, true, &params);
        }
        let w_before = syn.weight;
        // Pre spike triggers LTD via accumulated post-trace
        syn.step(true, false, &params);
        assert!(syn.weight < w_before, "LTD must decrease weight");
    }

    // ── RewardStdpSynapse tests ─────────────────────────────────

    #[test]
    fn rstdp_positive_reward_potentiates() {
        let mut syn = RewardStdpSynapse::new(0.5, 0.0, 1.0);
        for _ in 0..10 {
            syn.step(true, true);
        }
        let w_before = syn.weight;
        syn.apply_reward(1.0);
        assert!(syn.weight > w_before);
    }

    #[test]
    fn rstdp_negative_reward_depresses() {
        let mut syn = RewardStdpSynapse::new(0.5, 0.0, 1.0);
        for _ in 0..10 {
            syn.step(true, true);
        }
        let w_before = syn.weight;
        syn.apply_reward(-1.0);
        assert!(syn.weight < w_before);
    }

    #[test]
    fn rstdp_weight_bounded() {
        let mut syn = RewardStdpSynapse::new(0.5, 0.0, 1.0);
        for _ in 0..100 {
            syn.step(true, true);
            syn.apply_reward(10.0);
        }
        assert!(syn.weight <= 1.0);
        assert!(syn.weight >= 0.0);
    }

    // ── StaticSynapse tests ───────────────────────────────────────

    #[test]
    fn static_excitatory() {
        let syn = StaticSynapse::new(0.5, true);
        assert!((syn.transmit(true) - 0.5).abs() < 1e-12);
        assert!((syn.transmit(false)).abs() < 1e-12);
    }

    #[test]
    fn static_inhibitory() {
        let syn = StaticSynapse::new(0.5, false);
        assert!((syn.transmit(true) + 0.5).abs() < 1e-12);
    }

    #[test]
    fn weight_stays_in_bounds() {
        let mut syn = StdpSynapse::new(0, 16, 8);
        let params = default_params();
        for _ in 0..200 {
            syn.step(true, false, &params);
        }
        assert!(syn.weight >= params.w_min, "weight below w_min");
        assert!(syn.weight <= params.w_max, "weight above w_max");

        let mut syn2 = StdpSynapse::new(255, 16, 8);
        for _ in 0..200 {
            syn2.step(false, true, &params);
        }
        assert!(syn2.weight >= params.w_min);
        assert!(syn2.weight <= params.w_max);
    }

    // ── TripletStdpSynapse tests ────────────────────────────────

    #[test]
    fn triplet_ltp_pre_then_post() {
        let mut syn = TripletStdpSynapse::new(0.5, 0.0, 1.0);
        syn.step(true, false); // pre
        syn.step(false, true); // post → LTP
        assert!(syn.weight > 0.5, "Pre-then-post must potentiate");
    }

    #[test]
    fn triplet_ltd_post_then_pre() {
        let mut syn = TripletStdpSynapse::new(0.5, 0.0, 1.0);
        syn.step(false, true); // post
        syn.step(true, false); // pre → LTD
        assert!(syn.weight < 0.5, "Post-then-pre must depress");
    }

    #[test]
    fn triplet_bounded() {
        let mut syn = TripletStdpSynapse::new(0.5, 0.0, 1.0);
        for _ in 0..1000 {
            syn.step(true, true);
        }
        assert!(syn.weight >= 0.0 && syn.weight <= 1.0);
    }

    #[test]
    fn triplet_slow_trace_enhances() {
        // With o2 (slow post trace) pre-loaded, LTP should be stronger.
        let mut syn1 = TripletStdpSynapse::new(0.5, 0.0, 1.0);
        let mut syn2 = TripletStdpSynapse::new(0.5, 0.0, 1.0);
        // syn2 gets extra post spikes to build o2 (slow trace).
        for _ in 0..5 {
            syn2.step(false, true);
        }
        // Let fast traces (r1, o1) decay away while o2 (slow, tau_y=125) persists.
        for _ in 0..200 {
            syn2.step(false, false);
        }
        // Both start from same weight.
        syn1.weight = 0.5;
        syn2.weight = 0.5;
        // Single pre→post pair on both.
        syn1.step(true, false);
        syn1.step(false, true);
        syn2.step(true, false);
        syn2.step(false, true);
        // syn2 should get stronger LTP due to a3_plus * o2 contribution.
        assert!(
            syn2.weight >= syn1.weight,
            "Triplet o2 trace should enhance LTP: syn2={:.6} >= syn1={:.6}",
            syn2.weight,
            syn1.weight
        );
    }

    // ── ShortTermPlasticitySynapse tests ────────────────────────

    #[test]
    fn stp_depressing_decreases_psc() {
        let mut syn = ShortTermPlasticitySynapse::new_depressing();
        let psc1 = syn.step(true);
        let psc2 = syn.step(true);
        assert!(
            psc2 < psc1,
            "Depression: 2nd PSC < 1st: {psc2:.4} < {psc1:.4}"
        );
    }

    #[test]
    fn stp_facilitating_increases_psc() {
        let mut syn = ShortTermPlasticitySynapse::new_facilitating();
        let psc1 = syn.step(true);
        let psc2 = syn.step(true);
        assert!(
            psc2 > psc1,
            "Facilitation: 2nd PSC > 1st: {psc2:.4} > {psc1:.4}"
        );
    }

    #[test]
    fn stp_recovers_after_silence() {
        let mut syn = ShortTermPlasticitySynapse::new_depressing();
        syn.step(true);
        syn.step(true);
        let depleted = syn.step(true);
        // Wait for recovery.
        for _ in 0..500 {
            syn.step(false);
        }
        let recovered = syn.step(true);
        assert!(
            recovered > depleted,
            "Recovery: {recovered:.4} > {depleted:.4}"
        );
    }

    #[test]
    fn stp_no_spike_no_current() {
        let mut syn = ShortTermPlasticitySynapse::new_depressing();
        assert_eq!(syn.step(false), 0.0);
    }

    // ── DopamineStdpSynapse tests ──────────────────────────────

    #[test]
    fn da_stdp_reward_potentiates() {
        let mut syn = DopamineStdpSynapse::new(0.5, 0.0, 1.0);
        // Build eligibility: pre→post.
        for _ in 0..20 {
            syn.step(true, false, 0.0);
            syn.step(false, true, 0.0);
        }
        let w_before = syn.weight;
        // Deliver reward.
        for _ in 0..100 {
            syn.step(false, false, 1.0);
        }
        assert!(
            syn.weight > w_before,
            "Reward should potentiate: {:.4} > {:.4}",
            syn.weight,
            w_before
        );
    }

    #[test]
    fn da_stdp_no_reward_no_change() {
        let mut syn = DopamineStdpSynapse::new(0.5, 0.0, 1.0);
        // Build eligibility but no reward.
        for _ in 0..100 {
            syn.step(true, false, 0.0);
            syn.step(false, true, 0.0);
        }
        // Weight might change slightly due to e*DA ≈ 0 (DA decays to 0).
        assert!(
            (syn.weight - 0.5).abs() < 0.01,
            "Without reward, weight should stay near initial: {:.4}",
            syn.weight
        );
    }

    #[test]
    fn da_stdp_bounded() {
        let mut syn = DopamineStdpSynapse::new(0.5, 0.0, 1.0);
        for _ in 0..1000 {
            syn.step(true, true, 10.0);
        }
        assert!(syn.weight >= 0.0 && syn.weight <= 1.0);
    }
}
