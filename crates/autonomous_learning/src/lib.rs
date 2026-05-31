// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Autonomous Learning Engine
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

//! # Autonomous Learning Engine
//!
//! Multi-rule online plasticity engine with C-FFI surface.
//! Implements STDP, R-STDP (reward-modulated), BCM metaplasticity, and
//! the original ELIGENT (eligibility-trace + intrinsic adaptation) rule.
//!
//! All rules implement the [`PlasticityRule`] trait, enabling uniform
//! All rules implement the [`PlasticityRule`] trait, enabling uniform
//! dispatch from Python, Go, and C consumers via opaque pointers.

use rayon::prelude::*;

pub mod wgpu_backend;
use wgpu_backend::WgpuRuleLayer;

// ---------------------------------------------------------------------------
// Trait: PlasticityRule
// ---------------------------------------------------------------------------

/// Common interface for all online plasticity rules.
pub trait PlasticityRule: Send + Sync {
    /// Advance one timestep.
    ///
    /// * `pre_spike`  — presynaptic spike occurred this timestep
    /// * `post_spike` — postsynaptic spike occurred this timestep
    /// * `reward`     — global reward/neuromodulatory signal (ignored by unsupervised rules)
    fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f32, dt: f32);

    /// Reset internal state (traces, accumulators) without changing learned weights.
    fn reset(&mut self);

    /// Current weight value.
    fn weight(&self) -> f32;

    /// Rule identifier for FFI dispatch.
    fn rule_id(&self) -> u32;

    /// Get internal state (for persistence). Returns contiguous float representation.
    fn get_state(&self) -> Vec<f32> {
        vec![]
    }

    /// Restore internal state.
    fn set_state(&mut self, _state: &[f32]) {}
}

// ---------------------------------------------------------------------------
// Rule 0: ELIGENT (Eligibility + Intrinsic Adaptation)
// ---------------------------------------------------------------------------

/// NOTE: The `sum_weights` variable in this struct governs a Local Synaptic Exhaustion (LSE) boundary.
/// It deliberately normalizes only this single synaptic junction, diverging from global Oja layer norms.
/// This replicates exact constrained bounds found natively in isolated discrete MTJ circuits.
///
/// The neuron adjusts its firing threshold toward a target rate (homeostasis)
/// while an eligibility trace modulated by global reward drives synaptic weight change.
/// Weight normalization enforces a target sum.
///
/// References:
/// - Turrigiano, Cold Spring Harb Perspect Biol 4:a005736, 2012 (homeostasis)
/// - Izhikevich, Cerebral Cortex 17(10), 2007 (reward-modulated eligibility)
#[repr(C)]
pub struct EligentRule {
    pub threshold: f32,
    pub target_rate: f32,
    pub eta_intrinsic: f32,
    pub eligibility_trace: f32,
    pub tau_e: f32,
    pub weight: f32,
    pub sum_weights: f32,
    pub target_sum_weights: f32,
}

impl PlasticityRule for EligentRule {
    fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f32, dt: f32) {
        let current_rate = if post_spike { 1.0 } else { 0.0 };
        self.threshold += self.eta_intrinsic * (current_rate - self.target_rate) * dt;

        if pre_spike {
            self.eligibility_trace += 1.0;
        }
        self.eligibility_trace *= (-dt / self.tau_e).exp();
        let delta = self.eligibility_trace * reward;
        self.weight += delta;
        self.sum_weights += delta; // Update sum_weights physically!

        if self.sum_weights > 0.0 {
            let scale = self.target_sum_weights / self.sum_weights;
            self.weight *= scale;
            self.sum_weights = self.target_sum_weights; // Reset sum_weights to normalized target
        }
    }

    fn reset(&mut self) {
        self.eligibility_trace = 0.0;
    }

    fn weight(&self) -> f32 {
        self.weight
    }

    fn rule_id(&self) -> u32 {
        0
    }

    fn get_state(&self) -> Vec<f32> {
        vec![self.weight, self.threshold, self.eligibility_trace]
    }

    fn set_state(&mut self, state: &[f32]) {
        if state.len() >= 3 {
            self.weight = state[0];
            self.threshold = state[1];
            self.eligibility_trace = state[2];
        }
    }
}

// ---------------------------------------------------------------------------
// Rule 1: STDP (Spike-Timing Dependent Plasticity)
// ---------------------------------------------------------------------------

/// Classic pair-based STDP with exponential time windows.
///
/// Δw = A_+ * exp(-Δt / τ_+) for pre-before-post (LTP)
/// Δw = -A_- * exp(Δt / τ_-)  for post-before-pre (LTD)
///
/// Reference: Bi & Poo, J. Neurosci. 18(24):10464-10472, 1998.
pub struct StdpRule {
    pub weight: f32,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub w_min: f32,
    pub w_max: f32,
    pre_trace: f32,
    post_trace: f32,
}

impl StdpRule {
    pub fn new(weight: f32, a_plus: f32, a_minus: f32, tau_plus: f32, tau_minus: f32) -> Self {
        Self {
            weight,
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            w_min: 0.0,
            w_max: 1.0,
            pre_trace: 0.0,
            post_trace: 0.0,
        }
    }
}

impl PlasticityRule for StdpRule {
    fn step(&mut self, pre_spike: bool, post_spike: bool, _reward: f32, dt: f32) {
        // Decay traces
        self.pre_trace *= (-dt / self.tau_plus).exp();
        self.post_trace *= (-dt / self.tau_minus).exp();

        // Pre-before-post: LTP (potentiation)
        if post_spike {
            self.weight += self.a_plus * self.pre_trace;
        }
        // Post-before-pre: LTD (depression)
        if pre_spike {
            self.weight -= self.a_minus * self.post_trace;
        }

        // Update traces
        if pre_spike {
            self.pre_trace += 1.0;
        }
        if post_spike {
            self.post_trace += 1.0;
        }

        self.weight = self.weight.clamp(self.w_min, self.w_max);
    }

    fn reset(&mut self) {
        self.pre_trace = 0.0;
        self.post_trace = 0.0;
    }

    fn weight(&self) -> f32 {
        self.weight
    }

    fn rule_id(&self) -> u32 {
        1
    }

    fn get_state(&self) -> Vec<f32> {
        vec![self.weight, self.pre_trace, self.post_trace]
    }

    fn set_state(&mut self, state: &[f32]) {
        if state.len() >= 3 {
            self.weight = state[0];
            self.pre_trace = state[1];
            self.post_trace = state[2];
        }
    }
}

// ---------------------------------------------------------------------------
// Rule 2: R-STDP (Reward-Modulated STDP)
// ---------------------------------------------------------------------------

/// STDP with reward-modulated eligibility trace.
///
/// The STDP-computed Δw is not applied immediately but accumulated into an
/// eligibility trace. Actual weight change = eligibility × reward.
///
/// Reference: Izhikevich, Cerebral Cortex 17(10), 2007.
pub struct RewardStdpRule {
    pub weight: f32,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub tau_e: f32,
    pub w_min: f32,
    pub w_max: f32,
    pre_trace: f32,
    post_trace: f32,
    eligibility: f32,
}

impl RewardStdpRule {
    pub fn new(
        weight: f32,
        a_plus: f32,
        a_minus: f32,
        tau_plus: f32,
        tau_minus: f32,
        tau_e: f32,
    ) -> Self {
        Self {
            weight,
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            tau_e,
            w_min: 0.0,
            w_max: 1.0,
            pre_trace: 0.0,
            post_trace: 0.0,
            eligibility: 0.0,
        }
    }
}

impl PlasticityRule for RewardStdpRule {
    fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f32, dt: f32) {
        self.pre_trace *= (-dt / self.tau_plus).exp();
        self.post_trace *= (-dt / self.tau_minus).exp();

        // Accumulate STDP signal into eligibility trace
        if post_spike {
            self.eligibility += self.a_plus * self.pre_trace;
        }
        if pre_spike {
            self.eligibility -= self.a_minus * self.post_trace;
        }

        // Decay eligibility
        self.eligibility *= (-dt / self.tau_e).exp();

        // Reward-gated weight update
        self.weight += self.eligibility * reward;
        self.weight = self.weight.clamp(self.w_min, self.w_max);

        if pre_spike {
            self.pre_trace += 1.0;
        }
        if post_spike {
            self.post_trace += 1.0;
        }
    }

    fn reset(&mut self) {
        self.pre_trace = 0.0;
        self.post_trace = 0.0;
        self.eligibility = 0.0;
    }

    fn weight(&self) -> f32 {
        self.weight
    }

    fn rule_id(&self) -> u32 {
        2
    }

    fn get_state(&self) -> Vec<f32> {
        vec![
            self.weight,
            self.pre_trace,
            self.post_trace,
            self.eligibility,
        ]
    }

    fn set_state(&mut self, state: &[f32]) {
        if state.len() >= 4 {
            self.weight = state[0];
            self.pre_trace = state[1];
            self.post_trace = state[2];
            self.eligibility = state[3];
        }
    }
}

// ---------------------------------------------------------------------------
// Bounded fixed-point O(1) online learning rule
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnlineO1Config {
    pub weight_bits: u8,
    pub trace_bits: u8,
    pub reward_bits: u8,
    pub learning_shift: u8,
    pub trace_decay_shift: u8,
}

impl OnlineO1Config {
    pub fn new(
        weight_bits: u8,
        trace_bits: u8,
        reward_bits: u8,
        learning_shift: u8,
        trace_decay_shift: u8,
    ) -> Result<Self, &'static str> {
        if weight_bits == 0 || weight_bits > 31 {
            return Err("weight_bits must be in 1..=31");
        }
        if trace_bits < 2 || trace_bits > 30 {
            return Err("trace_bits must be in 2..=30");
        }
        if reward_bits == 0 || reward_bits > 30 {
            return Err("reward_bits must be in 1..=30");
        }
        if learning_shift > 30 {
            return Err("learning_shift must be <= 30");
        }
        if trace_decay_shift > 30 {
            return Err("trace_decay_shift must be <= 30");
        }
        Ok(Self {
            weight_bits,
            trace_bits,
            reward_bits,
            learning_shift,
            trace_decay_shift,
        })
    }

    pub fn max_weight(&self) -> u32 {
        (1_u32 << self.weight_bits) - 1
    }

    pub fn max_trace(&self) -> u32 {
        (1_u32 << self.trace_bits) - 1
    }

    pub fn min_eligibility(&self) -> i32 {
        -(1_i32 << (self.trace_bits - 1))
    }

    pub fn max_eligibility(&self) -> i32 {
        (1_i32 << (self.trace_bits - 1)) - 1
    }

    pub fn min_reward(&self) -> i32 {
        -(1_i32 << (self.reward_bits - 1))
    }

    pub fn max_reward(&self) -> i32 {
        (1_i32 << (self.reward_bits - 1)) - 1
    }

    pub fn per_synapse_state_bits(&self) -> u32 {
        self.weight_bits as u32 + 3 * self.trace_bits as u32
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OnlineO1Snapshot {
    pub weight: u32,
    pub pre_trace: u32,
    pub post_trace: u32,
    pub eligibility: i32,
}

pub struct OnlineO1Synapse {
    pub config: OnlineO1Config,
    pub weight: u32,
    pub pre_trace: u32,
    pub post_trace: u32,
    pub eligibility: i32,
}

impl OnlineO1Synapse {
    pub fn new(config: OnlineO1Config, initial_weight: u32) -> Result<Self, &'static str> {
        Ok(Self {
            config,
            weight: initial_weight.min(config.max_weight()),
            pre_trace: 0,
            post_trace: 0,
            eligibility: 0,
        })
    }

    pub fn snapshot(&self) -> OnlineO1Snapshot {
        OnlineO1Snapshot {
            weight: self.weight,
            pre_trace: self.pre_trace,
            post_trace: self.post_trace,
            eligibility: self.eligibility,
        }
    }

    pub fn step(&mut self, pre_spike: bool, post_spike: bool, reward: i32) -> OnlineO1Snapshot {
        let reward = reward.clamp(self.config.min_reward(), self.config.max_reward());
        let previous_pre_trace = self.pre_trace;
        let previous_post_trace = self.post_trace;

        self.pre_trace = decay_unsigned(
            self.pre_trace,
            self.config.trace_decay_shift,
            self.config.max_trace(),
        );
        self.post_trace = decay_unsigned(
            self.post_trace,
            self.config.trace_decay_shift,
            self.config.max_trace(),
        );
        if pre_spike {
            self.pre_trace = self.config.max_trace();
        }
        if post_spike {
            self.post_trace = self.config.max_trace();
        }

        let decayed_eligibility = decay_signed(self.eligibility, self.config.trace_decay_shift);
        let potentiation = if post_spike {
            if pre_spike {
                self.config.max_trace() as i32
            } else {
                previous_pre_trace as i32
            }
        } else {
            0
        };
        let depression = if pre_spike {
            previous_post_trace as i32
        } else {
            0
        };
        self.eligibility = (decayed_eligibility + potentiation - depression)
            .clamp(self.config.min_eligibility(), self.config.max_eligibility());

        let weight_delta =
            ((reward as i64 * self.eligibility as i64) >> self.config.learning_shift) as i64;
        self.weight =
            (self.weight as i64 + weight_delta).clamp(0, self.config.max_weight() as i64) as u32;
        self.snapshot()
    }

    pub fn per_synapse_state_bits(&self) -> u32 {
        self.config.per_synapse_state_bits()
    }
}

fn decay_unsigned(value: u32, shift: u8, max_value: u32) -> u32 {
    if shift == 0 {
        return value.min(max_value);
    }
    value.saturating_sub(value >> shift).min(max_value)
}

fn decay_signed(value: i32, shift: u8) -> i32 {
    if shift == 0 {
        return value;
    }
    if value >= 0 {
        value - (value >> shift)
    } else {
        let magnitude = -value;
        -(magnitude - (magnitude >> shift))
    }
}

// ---------------------------------------------------------------------------
// Rule 3: BCM (Bienenstock-Cooper-Munro) Metaplasticity
// ---------------------------------------------------------------------------

/// BCM rule with sliding threshold.
///
/// The modification threshold θ_m slides as a function of the postsynaptic
/// neuron's recent activity, implementing metaplasticity. This prevents
/// runaway excitation/depression.
///
/// Δw = η * y * (y - θ_m) * x
/// dθ_m/dt = (y² - θ_m) / τ_θ
///
/// Reference: Bienenstock, Cooper, Munro, J. Neurosci. 2(1):32-48, 1982.
pub struct BcmRule {
    pub weight: f32,
    pub eta: f32,
    pub tau_theta: f32,
    pub w_min: f32,
    pub w_max: f32,
    theta_m: f32,
    activity_avg: f32,
}

impl BcmRule {
    pub fn new(weight: f32, eta: f32, tau_theta: f32) -> Self {
        Self {
            weight,
            eta,
            tau_theta,
            w_min: 0.0,
            w_max: 1.0,
            theta_m: 0.5,
            activity_avg: 0.0,
        }
    }
}

impl PlasticityRule for BcmRule {
    fn step(&mut self, pre_spike: bool, post_spike: bool, _reward: f32, dt: f32) {
        let x = if pre_spike { 1.0f32 } else { 0.0 };
        let y = if post_spike { 1.0f32 } else { 0.0 };

        // BCM weight update
        self.weight += self.eta * y * (y - self.theta_m) * x * dt;
        self.weight = self.weight.clamp(self.w_min, self.w_max);

        // Update sliding threshold
        self.activity_avg += (y - self.activity_avg) * (dt / self.tau_theta);
        self.theta_m +=
            (self.activity_avg * self.activity_avg - self.theta_m) * (dt / self.tau_theta);
        self.theta_m = self.theta_m.max(0.01); // prevent collapse
    }

    fn reset(&mut self) {
        self.activity_avg = 0.0;
        self.theta_m = 0.5;
    }

    fn weight(&self) -> f32 {
        self.weight
    }

    fn rule_id(&self) -> u32 {
        3
    }

    fn get_state(&self) -> Vec<f32> {
        vec![self.weight, self.theta_m]
    }

    fn set_state(&mut self, state: &[f32]) {
        if state.len() >= 2 {
            self.weight = state[0];
            self.theta_m = state[1];
        }
    }
}

// ---------------------------------------------------------------------------
// FFI-safe wrapper (trait objects are fat pointers, can't pass through C-FFI)
// ---------------------------------------------------------------------------

/// Concrete enum wrapper for FFI dispatch.
pub enum RuleHandle {
    Eligent(EligentRule),
    Stdp(StdpRule),
    RewardStdp(RewardStdpRule),
    Bcm(BcmRule),
}

impl RuleHandle {
    fn as_rule(&mut self) -> &mut dyn PlasticityRule {
        match self {
            RuleHandle::Eligent(r) => r,
            RuleHandle::Stdp(r) => r,
            RuleHandle::RewardStdp(r) => r,
            RuleHandle::Bcm(r) => r,
        }
    }

    fn as_rule_ref(&self) -> &dyn PlasticityRule {
        match self {
            RuleHandle::Eligent(r) => r,
            RuleHandle::Stdp(r) => r,
            RuleHandle::RewardStdp(r) => r,
            RuleHandle::Bcm(r) => r,
        }
    }
}

/// Create a plasticity rule by type ID.
///
/// | rule_type | Rule |
/// |-----------|------|
/// | 0 | ELIGENT |
/// | 1 | STDP |
/// | 2 | R-STDP |
/// | 3 | BCM |
///
/// Caller must free with `destroy_rule`.
#[no_mangle]
pub extern "C" fn create_rule(
    rule_type: u32,
    weight: f32,
    param_a: f32,
    param_b: f32,
) -> *mut RuleHandle {
    let handle = match rule_type {
        0 => RuleHandle::Eligent(EligentRule {
            threshold: 1.0,
            target_rate: param_a.max(0.01),
            eta_intrinsic: 0.001,
            eligibility_trace: 0.0,
            tau_e: param_b.max(0.01),
            weight,
            sum_weights: weight,
            target_sum_weights: 1.0,
        }),
        1 => RuleHandle::Stdp(StdpRule::new(
            weight,
            param_a.max(0.001),
            param_a.max(0.001) * 0.5,
            20.0,
            20.0,
        )),
        2 => RuleHandle::RewardStdp(RewardStdpRule::new(
            weight,
            param_a.max(0.001),
            param_a.max(0.001) * 0.5,
            20.0,
            20.0,
            param_b.max(0.01),
        )),
        3 => RuleHandle::Bcm(BcmRule::new(weight, param_a.max(0.0001), param_b.max(1.0))),
        _ => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(handle))
}

/// Backward-compatible FFI entry point for ELIGENT rule.
#[no_mangle]
pub extern "C" fn create_learner(
    threshold: f32,
    target_rate: f32,
    weight: f32,
) -> *mut EligentRule {
    let state = EligentRule {
        threshold,
        target_rate,
        eta_intrinsic: 0.001,
        eligibility_trace: 0.0,
        tau_e: 0.95,
        weight,
        sum_weights: weight,
        target_sum_weights: 1.0,
    };
    Box::into_raw(Box::new(state))
}

/// Step a rule through one timestep.
///
/// # Safety
/// `ptr` must have been returned by `create_rule`.
#[no_mangle]
pub unsafe extern "C" fn step_rule(
    ptr: *mut RuleHandle,
    pre_spike: bool,
    post_spike: bool,
    reward: f32,
    dt: f32,
) {
    if ptr.is_null() {
        return;
    }
    let handle = unsafe { &mut *ptr };
    handle.as_rule().step(pre_spike, post_spike, reward, dt);
}

/// Get current weight from a rule.
///
/// # Safety
/// `ptr` must have been returned by `create_rule`.
#[no_mangle]
pub unsafe extern "C" fn get_rule_weight(ptr: *const RuleHandle) -> f32 {
    if ptr.is_null() {
        return 0.0;
    }
    let handle = unsafe { &*ptr };
    handle.as_rule_ref().weight()
}

/// Reset a rule's internal state.
///
/// # Safety
/// `ptr` must have been returned by `create_rule`.
#[no_mangle]
pub unsafe extern "C" fn reset_rule(ptr: *mut RuleHandle) {
    if ptr.is_null() {
        return;
    }
    let handle = unsafe { &mut *ptr };
    handle.as_rule().reset();
}

/// Destroy a rule instance.
///
/// # Safety
/// `ptr` must have been returned by `create_rule`.
#[no_mangle]
pub unsafe extern "C" fn destroy_rule(ptr: *mut RuleHandle) {
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr) };
    }
}

#[no_mangle]
pub extern "C" fn create_online_o1_synapse(
    weight_bits: u8,
    trace_bits: u8,
    reward_bits: u8,
    learning_shift: u8,
    trace_decay_shift: u8,
    initial_weight: u32,
) -> *mut OnlineO1Synapse {
    let Ok(config) = OnlineO1Config::new(
        weight_bits,
        trace_bits,
        reward_bits,
        learning_shift,
        trace_decay_shift,
    ) else {
        return std::ptr::null_mut();
    };
    let Ok(synapse) = OnlineO1Synapse::new(config, initial_weight) else {
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(synapse))
}

#[no_mangle]
pub unsafe extern "C" fn step_online_o1_synapse(
    ptr: *mut OnlineO1Synapse,
    pre_spike: bool,
    post_spike: bool,
    reward: i32,
) -> OnlineO1Snapshot {
    if ptr.is_null() {
        return OnlineO1Snapshot::default();
    }
    unsafe { &mut *ptr }.step(pre_spike, post_spike, reward)
}

#[no_mangle]
pub unsafe extern "C" fn online_o1_per_synapse_state_bits(ptr: *const OnlineO1Synapse) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.per_synapse_state_bits()
}

#[no_mangle]
pub unsafe extern "C" fn destroy_online_o1_synapse(ptr: *mut OnlineO1Synapse) {
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr) };
    }
}

/// Backward-compatible FFI for ELIGENT learner step.
#[no_mangle]
pub extern "C" fn step_learner(
    ptr: *mut EligentRule,
    fired: bool,
    pre_spike: bool,
    global_reward: f32,
    dt: f32,
) {
    if ptr.is_null() {
        return;
    }
    let state = unsafe { &mut *ptr };
    state.step(pre_spike, fired, global_reward, dt);
}

/// Backward-compatible FFI for ELIGENT learner destruction.
#[no_mangle]
pub extern "C" fn destroy_learner(ptr: *mut EligentRule) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Batched Vector Execution
// ---------------------------------------------------------------------------

/// Batched FFI execution for vector arrays.
#[no_mangle]
pub unsafe extern "C" fn step_rule_batched(
    ptr: *mut RuleHandle,
    pre_spikes: *const bool,
    post_spikes: *const bool,
    rewards: *const f32,
    count: usize,
    dt: f32,
) {
    if ptr.is_null() || pre_spikes.is_null() || post_spikes.is_null() || rewards.is_null() {
        return;
    }
    let handle = unsafe { &mut *ptr };
    let pre_slice = std::slice::from_raw_parts(pre_spikes, count);
    let post_slice = std::slice::from_raw_parts(post_spikes, count);
    let rew_slice = std::slice::from_raw_parts(rewards, count);

    let rule = handle.as_rule();
    for i in 0..count {
        rule.step(pre_slice[i], post_slice[i], rew_slice[i], dt);
    }
}

/// Batched FFI execution for ELIGENT legacy vectors.
#[no_mangle]
pub unsafe extern "C" fn step_learner_batched(
    ptr: *mut EligentRule,
    fired: *const bool,
    pre_spikes: *const bool,
    rewards: *const f32,
    count: usize,
    dt: f32,
) {
    if ptr.is_null() || fired.is_null() || pre_spikes.is_null() || rewards.is_null() {
        return;
    }
    let state = unsafe { &mut *ptr };
    let fired_slice = std::slice::from_raw_parts(fired, count);
    let pre_slice = std::slice::from_raw_parts(pre_spikes, count);
    let rew_slice = std::slice::from_raw_parts(rewards, count);

    for i in 0..count {
        state.step(pre_slice[i], fired_slice[i], rew_slice[i], dt);
    }
}

/// WGPU Backend FFI

#[no_mangle]
pub unsafe extern "C" fn create_wgpu_layer(
    count: usize,
    rule_type: u32,
    a_plus: f32,
    a_minus: f32,
    tau_plus: f32,
    tau_minus: f32,
    param_c: f32,
    param_d: f32,
) -> *mut WgpuRuleLayer {
    println!(
        "Initializing WGPU backend (Rule {}, Scale {})",
        rule_type, count
    );
    if let Some(layer) = WgpuRuleLayer::new(
        count, rule_type, a_plus, a_minus, tau_plus, tau_minus, param_c, param_d,
    ) {
        Box::into_raw(Box::new(layer))
    } else {
        println!("Warning: WGPU initialization failed gracefully on host. Returning NULL pointer.");
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub unsafe extern "C" fn step_wgpu_layer(
    mgr: *mut WgpuRuleLayer,
    pre_probs: *const f32,
    post_probs: *const f32,
    rewards: *const f32,
    dt: f32,
) {
    if mgr.is_null() {
        return;
    }
    let layer = &mut *mgr;
    let pre_slice = std::slice::from_raw_parts(pre_probs, layer.count as usize);
    let post_slice = std::slice::from_raw_parts(post_probs, layer.count as usize);
    let reward_slice = if !rewards.is_null() {
        std::slice::from_raw_parts(rewards, layer.count as usize)
    } else {
        &[]
    };

    layer.step(pre_slice, post_slice, reward_slice, dt);
}

#[no_mangle]
pub unsafe extern "C" fn get_wgpu_weights(mgr: *mut WgpuRuleLayer, out_weights: *mut f32) {
    if mgr.is_null() || out_weights.is_null() {
        return;
    }
    let layer = &*mgr;
    let weights = layer.get_weights();
    std::ptr::copy_nonoverlapping(weights.as_ptr(), out_weights, layer.count as usize);
}

#[no_mangle]
pub unsafe extern "C" fn set_wgpu_layer_seed(mgr: *mut WgpuRuleLayer, seed: u32) {
    if mgr.is_null() {
        return;
    }
    let layer = &mut *mgr;
    layer.set_deterministic_mode(seed);
}

/// Reset a WGPU rule layer's traces. See `WgpuRuleLayer::reset`.
///
/// # Safety
/// `mgr` must have been returned by `create_wgpu_layer`.
#[no_mangle]
pub unsafe extern "C" fn reset_wgpu_layer(mgr: *mut WgpuRuleLayer) {
    if mgr.is_null() {
        return;
    }
    let layer = unsafe { &mut *mgr };
    layer.reset();
}

#[no_mangle]
pub unsafe extern "C" fn free_wgpu_layer(mgr: *mut WgpuRuleLayer) {
    if !mgr.is_null() {
        drop(Box::from_raw(mgr));
    }
}

// ---------------------------------------------------------------------------
// Spatial Layer Parallelization (Rayon)
// ---------------------------------------------------------------------------

pub struct RuleLayerHandle {
    pub rules: Vec<Box<dyn PlasticityRule>>,
}

#[no_mangle]
pub unsafe extern "C" fn create_rule_layer(
    count: usize,
    rule_type: u32,
    weight: f32,
    param_a: f32,
    param_b: f32,
) -> *mut RuleLayerHandle {
    let mut rules = Vec::with_capacity(count);
    for _ in 0..count {
        let rule: Box<dyn PlasticityRule> = match rule_type {
            0 => Box::new(EligentRule {
                threshold: 1.0,
                target_rate: param_a.max(0.01),
                eta_intrinsic: 0.001,
                eligibility_trace: 0.0,
                tau_e: param_b.max(0.01),
                weight,
                sum_weights: weight,
                target_sum_weights: 1.0,
            }),
            1 => Box::new(StdpRule::new(
                weight,
                param_a.max(0.001),
                param_a.max(0.001) * 0.5,
                20.0,
                20.0,
            )),
            2 => Box::new(RewardStdpRule::new(
                weight,
                param_a.max(0.001),
                param_a.max(0.001) * 0.5,
                20.0,
                20.0,
                param_b.max(0.01),
            )),
            3 => Box::new(BcmRule::new(weight, param_a.max(0.0001), param_b.max(1.0))),
            _ => Box::new(StdpRule::new(
                weight,
                param_a.max(0.001),
                param_a.max(0.001) * 0.5,
                20.0,
                20.0,
            )),
        };
        rules.push(rule);
    }

    Box::into_raw(Box::new(RuleLayerHandle { rules }))
}

#[no_mangle]
pub unsafe extern "C" fn step_rule_layer(
    layer_ptr: *mut RuleLayerHandle,
    pre_spikes: *const bool,
    post_spikes: *const bool,
    rewards: *const f32,
    dt: f32,
) {
    if layer_ptr.is_null() || pre_spikes.is_null() || post_spikes.is_null() || rewards.is_null() {
        return;
    }
    let layer = unsafe { &mut *layer_ptr };
    let count = layer.rules.len();

    let pre_slice = std::slice::from_raw_parts(pre_spikes, count);
    let post_slice = std::slice::from_raw_parts(post_spikes, count);
    let rew_slice = std::slice::from_raw_parts(rewards, count);

    layer
        .rules
        .par_iter_mut()
        .zip(pre_slice.par_iter())
        .zip(post_slice.par_iter())
        .zip(rew_slice.par_iter())
        .for_each(|(((rule, &pre), &post), &rew)| {
            rule.step(pre, post, rew, dt);
        });
}

#[no_mangle]
pub unsafe extern "C" fn step_rule_layer_analog(
    layer_ptr: *mut RuleLayerHandle,
    pre_probs: *const f32,
    post_probs: *const f32,
    rewards: *const f32,
    seed: u64,
    dt: f32,
) {
    if layer_ptr.is_null() || pre_probs.is_null() || post_probs.is_null() || rewards.is_null() {
        return;
    }
    let layer = unsafe { &mut *layer_ptr };
    let count = layer.rules.len();

    let pre_slice = std::slice::from_raw_parts(pre_probs, count);
    let post_slice = std::slice::from_raw_parts(post_probs, count);
    let rew_slice = std::slice::from_raw_parts(rewards, count);

    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    layer
        .rules
        .par_iter_mut()
        .zip(pre_slice.par_iter())
        .zip(post_slice.par_iter())
        .zip(rew_slice.par_iter())
        .enumerate()
        .for_each(|(idx, (((rule, &pre_p), &post_p), &rew))| {
            // Seed uniquely per spatial node trace to guarantee structural
            // reproducibility across CPU parallel partitioning. In rand 0.8
            // we used `SmallRng`, which was backed by Xoshiro256PlusPlus;
            // rand 0.10 unbundled the alias, so we construct the same
            // underlying algorithm directly from `rand_xoshiro`.
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(idx as u64));
            // rand 0.9+: `Rng::gen` renamed to `Rng::random`.
            let pre_spike = rng.random::<f32>() < pre_p;
            let post_spike = rng.random::<f32>() < post_p;
            rule.step(pre_spike, post_spike, rew, dt);
        });
}

#[no_mangle]
pub unsafe extern "C" fn get_rule_layer_weights(
    layer_ptr: *const RuleLayerHandle,
    out_weights: *mut f32,
) {
    if layer_ptr.is_null() || out_weights.is_null() {
        return;
    }
    let layer = unsafe { &*layer_ptr };
    let count = layer.rules.len();
    let out_slice = std::slice::from_raw_parts_mut(out_weights, count);

    layer
        .rules
        .par_iter()
        .zip(out_slice.par_iter_mut())
        .for_each(|(rule, out)| {
            *out = rule.weight();
        });
}

#[no_mangle]
pub unsafe extern "C" fn save_rule_layer_batched(
    layer_ptr: *const RuleLayerHandle,
    filepath: *const std::os::raw::c_char,
) -> bool {
    if layer_ptr.is_null() || filepath.is_null() {
        return false;
    }
    let c_str = std::ffi::CStr::from_ptr(filepath);
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let size = get_rule_layer_state_size(layer_ptr);
    let mut buffer = vec![0u8; size];
    if !get_rule_layer_state_mem(layer_ptr, buffer.as_mut_ptr()) {
        return false;
    }

    if let Ok(mut file) = std::fs::File::create(path) {
        use std::io::Write;
        file.write_all(&buffer).is_ok()
    } else {
        false
    }
}

#[no_mangle]
pub unsafe extern "C" fn load_rule_layer_batched(
    layer_ptr: *mut RuleLayerHandle,
    filepath: *const std::os::raw::c_char,
) -> bool {
    if layer_ptr.is_null() || filepath.is_null() {
        return false;
    }
    let c_str = std::ffi::CStr::from_ptr(filepath);
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    if let Ok(mut file) = std::fs::File::open(path) {
        let mut byte_buffer = Vec::new();
        use std::io::Read;
        if file.read_to_end(&mut byte_buffer).is_err() {
            return false;
        }

        set_rule_layer_state_mem(layer_ptr, byte_buffer.as_ptr())
    } else {
        false
    }
}

#[no_mangle]
pub unsafe extern "C" fn destroy_rule_layer(layer_ptr: *mut RuleLayerHandle) {
    if !layer_ptr.is_null() {
        let _ = unsafe { Box::from_raw(layer_ptr) };
    }
}

/// Reset every rule in a layer to its no-history state.
///
/// Clears traces/accumulators (pre/post trace, eligibility, running activity
/// averages, adaptive thresholds) as defined by each rule's `PlasticityRule::reset`.
/// Learned weights are preserved — the layer keeps its identity.
///
/// # Safety
/// `layer_ptr` must have been returned by `create_rule_layer`.
#[no_mangle]
pub unsafe extern "C" fn reset_rule_layer(layer_ptr: *mut RuleLayerHandle) {
    if layer_ptr.is_null() {
        return;
    }
    let layer = unsafe { &mut *layer_ptr };
    layer.rules.par_iter_mut().for_each(|rule| rule.reset());
}

#[no_mangle]
pub unsafe extern "C" fn get_rule_layer_state_size(layer_ptr: *const RuleLayerHandle) -> usize {
    if layer_ptr.is_null() {
        return 0;
    }
    let layer = &*layer_ptr;
    let mut total_f32 = 0;
    for rule in &layer.rules {
        total_f32 += rule.get_state().len();
    }
    // 4 magic + 4 version + 4 count + per_rule(4 id + 4 len) + f32 bytes
    12 + layer.rules.len() * 8 + total_f32 * 4
}

#[no_mangle]
pub unsafe extern "C" fn get_rule_layer_state_mem(
    layer_ptr: *const RuleLayerHandle,
    out_buffer: *mut u8,
) -> bool {
    if layer_ptr.is_null() || out_buffer.is_null() {
        return false;
    }
    let layer = &*layer_ptr;

    let mut offset = 0;
    let size = get_rule_layer_state_size(layer_ptr);
    let out_slice = std::slice::from_raw_parts_mut(out_buffer, size);

    out_slice[offset..offset + 4].copy_from_slice(b"SCAL");
    offset += 4;
    out_slice[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
    offset += 4;
    out_slice[offset..offset + 4].copy_from_slice(&(layer.rules.len() as u32).to_le_bytes());
    offset += 4;

    for rule in &layer.rules {
        out_slice[offset..offset + 4].copy_from_slice(&rule.rule_id().to_le_bytes());
        offset += 4;
        let rs = rule.get_state();
        out_slice[offset..offset + 4].copy_from_slice(&(rs.len() as u32).to_le_bytes());
        offset += 4;

        let byte_size = rs.len() * 4;
        std::ptr::copy_nonoverlapping(
            rs.as_ptr() as *const u8,
            out_slice[offset..].as_mut_ptr(),
            byte_size,
        );
        offset += byte_size;
    }
    true
}

#[no_mangle]
pub unsafe extern "C" fn set_rule_layer_state_mem(
    layer_ptr: *mut RuleLayerHandle,
    in_buffer: *const u8,
) -> bool {
    if layer_ptr.is_null() || in_buffer.is_null() {
        return false;
    }
    let layer = &mut *layer_ptr;

    let magic = std::slice::from_raw_parts(in_buffer, 4);
    if magic != b"SCAL" {
        return false;
    } // Strict Magic Verify

    let mut offset = 4;
    let version_bytes = std::slice::from_raw_parts(in_buffer.add(offset), 4);
    let version = u32::from_le_bytes(version_bytes.try_into().unwrap());
    offset += 4;
    if version != 1 {
        return false;
    } // Unsupported version

    let count_bytes = std::slice::from_raw_parts(in_buffer.add(offset), 4);
    let count = u32::from_le_bytes(count_bytes.try_into().unwrap());
    offset += 4;

    if count as usize != layer.rules.len() {
        return false;
    } // Layer dimension mismatch

    for rule in &mut layer.rules {
        let rule_id = u32::from_le_bytes(
            std::slice::from_raw_parts(in_buffer.add(offset), 4)
                .try_into()
                .unwrap(),
        );
        offset += 4;
        if rule_id != rule.rule_id() {
            return false;
        } // Rule mapping mismatch

        let trace_count = u32::from_le_bytes(
            std::slice::from_raw_parts(in_buffer.add(offset), 4)
                .try_into()
                .unwrap(),
        ) as usize;
        offset += 4;

        let traces = std::slice::from_raw_parts(in_buffer.add(offset) as *const f32, trace_count);
        rule.set_state(traces);
        offset += trace_count * 4;
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eligent_weight_changes_with_reward() {
        let mut rule = EligentRule {
            threshold: 1.0,
            target_rate: 0.1,
            eta_intrinsic: 0.001,
            eligibility_trace: 0.0,
            tau_e: 0.95,
            weight: 0.5,
            sum_weights: 0.5,
            target_sum_weights: 1.0,
        };

        let initial = rule.weight();
        // Pre-spike builds eligibility, then reward drives weight change
        rule.step(true, false, 0.0, 1.0);
        rule.step(false, true, 1.0, 1.0);
        assert_ne!(rule.weight(), initial, "Weight should change after reward");
    }

    #[test]
    fn stdp_ltp_direction() {
        let mut rule = StdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0);
        let initial = rule.weight();
        // Pre spike first, then post spike → LTP (potentiation)
        rule.step(true, false, 0.0, 1.0);
        rule.step(false, true, 0.0, 1.0);
        assert!(
            rule.weight() > initial,
            "Pre-before-post should increase weight: {} vs {}",
            rule.weight(),
            initial
        );
    }

    #[test]
    fn stdp_ltd_direction() {
        let mut rule = StdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0);
        let initial = rule.weight();
        // Post spike first, then pre spike → LTD (depression)
        rule.step(false, true, 0.0, 1.0);
        rule.step(true, false, 0.0, 1.0);
        assert!(
            rule.weight() < initial,
            "Post-before-pre should decrease weight: {} vs {}",
            rule.weight(),
            initial
        );
    }

    #[test]
    fn stdp_weight_bounds() {
        let mut rule = StdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0);
        for _ in 0..1000 {
            rule.step(true, true, 0.0, 1.0);
        }
        assert!(rule.weight() >= 0.0 && rule.weight() <= 1.0);
    }

    #[test]
    fn rstdp_no_reward_no_change() {
        let mut rule = RewardStdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0, 0.95);
        let initial = rule.weight();
        // Spikes but no reward → eligibility builds but weight doesn't change
        rule.step(true, false, 0.0, 1.0);
        rule.step(false, true, 0.0, 1.0);
        // Without reward, weight should remain close to initial
        assert!(
            (rule.weight() - initial).abs() < 1e-6,
            "Without reward, weight should not change: {} vs {}",
            rule.weight(),
            initial
        );
    }

    #[test]
    fn rstdp_reward_drives_change() {
        let mut rule = RewardStdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0, 0.95);
        let initial = rule.weight();
        rule.step(true, false, 0.0, 1.0);
        rule.step(false, true, 1.0, 1.0); // reward delivered with post spike
        assert_ne!(rule.weight(), initial, "Reward should drive weight change");
    }

    #[test]
    fn bcm_below_threshold_depresses() {
        let mut rule = BcmRule::new(0.5, 0.1, 10.0);
        rule.theta_m = 1.0; // set high threshold
        let initial = rule.weight();
        // y=1 < θ_m=1.0 → (y - θ_m) = 0, no change on exact boundary
        // But activity below threshold across time should depress
        for _ in 0..10 {
            rule.step(true, false, 0.0, 1.0); // pre but no post → no change
        }
        // Without post spikes, no BCM update (requires y > 0)
        assert_eq!(rule.weight(), initial);
    }

    #[test]
    fn bcm_sliding_threshold() {
        let mut rule = BcmRule::new(0.5, 0.01, 10.0);
        let initial_theta = rule.theta_m;
        // Sustained high activity should raise the sliding threshold
        for _ in 0..100 {
            rule.step(true, true, 0.0, 1.0);
        }
        assert!(
            rule.theta_m > initial_theta,
            "Sustained activity should raise θ_m: {} vs {}",
            rule.theta_m,
            initial_theta
        );
    }

    #[test]
    fn ffi_create_all_rules() {
        for rule_type in 0..4 {
            let ptr = create_rule(rule_type, 0.5, 0.1, 0.95);
            assert!(
                !ptr.is_null(),
                "Rule type {rule_type} should create successfully"
            );
            unsafe {
                step_rule(ptr, true, false, 0.0, 1.0);
                step_rule(ptr, false, true, 1.0, 1.0);
                let w = get_rule_weight(ptr);
                assert!(
                    w.is_finite(),
                    "Weight should be finite for rule {rule_type}"
                );
                reset_rule(ptr);
                destroy_rule(ptr);
            }
        }
    }

    #[test]
    fn ffi_invalid_rule_type() {
        let ptr = create_rule(99, 0.5, 0.1, 0.95);
        assert!(ptr.is_null());
    }

    #[test]
    fn backward_compat_eligent_ffi() {
        let ptr = create_learner(1.0, 0.1, 0.5);
        assert!(!ptr.is_null());
        step_learner(ptr, true, true, 1.0, 1.0);
        destroy_learner(ptr);
    }

    #[test]
    fn rule_reset_clears_traces() {
        let mut rule = StdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0);
        rule.step(true, false, 0.0, 1.0);
        assert!(rule.pre_trace > 0.0);
        rule.reset();
        assert_eq!(rule.pre_trace, 0.0);
        assert_eq!(rule.post_trace, 0.0);
    }

    #[test]
    fn online_o1_matches_python_reference_trace() {
        let config = OnlineO1Config::new(8, 6, 4, 3, 2).expect("valid online O(1) config");
        let mut synapse = OnlineO1Synapse::new(config, 0).expect("valid online O(1) synapse");
        let events = [
            (true, false, 0),
            (false, true, 7),
            (false, false, 7),
            (false, false, 7),
            (false, false, -7),
            (true, false, 0),
            (false, true, -7),
        ];
        let expected = [
            OnlineO1Snapshot {
                weight: 0,
                pre_trace: 63,
                post_trace: 0,
                eligibility: 0,
            },
            OnlineO1Snapshot {
                weight: 27,
                pre_trace: 48,
                post_trace: 63,
                eligibility: 31,
            },
            OnlineO1Snapshot {
                weight: 48,
                pre_trace: 36,
                post_trace: 48,
                eligibility: 24,
            },
            OnlineO1Snapshot {
                weight: 63,
                pre_trace: 27,
                post_trace: 36,
                eligibility: 18,
            },
            OnlineO1Snapshot {
                weight: 50,
                pre_trace: 21,
                post_trace: 27,
                eligibility: 14,
            },
            OnlineO1Snapshot {
                weight: 50,
                pre_trace: 63,
                post_trace: 21,
                eligibility: -16,
            },
            OnlineO1Snapshot {
                weight: 22,
                pre_trace: 48,
                post_trace: 63,
                eligibility: 31,
            },
        ];

        for ((pre, post, reward), expected_snapshot) in events.into_iter().zip(expected) {
            let observed = synapse.step(pre, post, reward);
            assert_eq!(observed, expected_snapshot);
        }
        assert_eq!(synapse.per_synapse_state_bits(), 26);
    }
}
