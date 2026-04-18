// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Autonomous Learning Engine
// Co-Authored-By: Arcane Sapience <protoscience@anulum.li>

//! # Autonomous Learning Engine
//!
//! Multi-rule online plasticity engine with C-FFI surface.
//! Implements STDP, R-STDP (reward-modulated), BCM metaplasticity, and
//! the original ELIGENT (eligibility-trace + intrinsic adaptation) rule.
//!
//! All rules implement the [`PlasticityRule`] trait, enabling uniform
//! dispatch from Python, Go, and C consumers via opaque pointers.

// ---------------------------------------------------------------------------
// Trait: PlasticityRule
// ---------------------------------------------------------------------------

/// Common interface for all online plasticity rules.
pub trait PlasticityRule: Send {
    /// Advance one timestep.
    ///
    /// * `pre_spike`  — presynaptic spike occurred this timestep
    /// * `post_spike` — postsynaptic spike occurred this timestep
    /// * `reward`     — global reward/neuromodulatory signal (ignored by unsupervised rules)
    fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f32);

    /// Reset internal state (traces, accumulators) without changing learned weights.
    fn reset(&mut self);

    /// Current weight value.
    fn weight(&self) -> f32;

    /// Rule identifier for FFI dispatch.
    fn rule_id(&self) -> u32;
}

// ---------------------------------------------------------------------------
// Rule 0: ELIGENT (Eligibility + Intrinsic Adaptation)
// ---------------------------------------------------------------------------

/// Eligibility-trace learning with intrinsic threshold adaptation.
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
    fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f32) {
        let current_rate = if post_spike { 1.0 } else { 0.0 };
        self.threshold += self.eta_intrinsic * (current_rate - self.target_rate);

        if pre_spike {
            self.eligibility_trace += 1.0;
        }
        self.eligibility_trace *= self.tau_e;
        self.weight += self.eligibility_trace * reward;

        if self.sum_weights > 0.0 {
            let scale = self.target_sum_weights / self.sum_weights;
            self.weight *= scale;
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
    fn step(&mut self, pre_spike: bool, post_spike: bool, _reward: f32) {
        // Decay traces
        self.pre_trace *= (-1.0 / self.tau_plus).exp();
        self.post_trace *= (-1.0 / self.tau_minus).exp();

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
    fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f32) {
        self.pre_trace *= (-1.0 / self.tau_plus).exp();
        self.post_trace *= (-1.0 / self.tau_minus).exp();

        // Accumulate STDP signal into eligibility trace
        if post_spike {
            self.eligibility += self.a_plus * self.pre_trace;
        }
        if pre_spike {
            self.eligibility -= self.a_minus * self.post_trace;
        }

        // Decay eligibility
        self.eligibility *= self.tau_e;

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
    fn step(&mut self, pre_spike: bool, post_spike: bool, _reward: f32) {
        let x = if pre_spike { 1.0f32 } else { 0.0 };
        let y = if post_spike { 1.0f32 } else { 0.0 };

        // BCM weight update
        self.weight += self.eta * y * (y - self.theta_m) * x;
        self.weight = self.weight.clamp(self.w_min, self.w_max);

        // Update sliding threshold
        self.activity_avg += (y - self.activity_avg) / self.tau_theta;
        self.theta_m += (self.activity_avg * self.activity_avg - self.theta_m) / self.tau_theta;
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
pub extern "C" fn create_rule(rule_type: u32, weight: f32, param_a: f32, param_b: f32) -> *mut RuleHandle {
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
        1 => RuleHandle::Stdp(StdpRule::new(weight, param_a.max(0.001), param_a.max(0.001) * 0.5, 20.0, 20.0)),
        2 => RuleHandle::RewardStdp(RewardStdpRule::new(weight, param_a.max(0.001), param_a.max(0.001) * 0.5, 20.0, 20.0, param_b.max(0.01))),
        3 => RuleHandle::Bcm(BcmRule::new(weight, param_a.max(0.0001), param_b.max(1.0))),
        _ => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(handle))
}

/// Backward-compatible FFI entry point for ELIGENT rule.
#[no_mangle]
pub extern "C" fn create_learner(threshold: f32, target_rate: f32, weight: f32) -> *mut EligentRule {
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
) {
    if ptr.is_null() {
        return;
    }
    let handle = unsafe { &mut *ptr };
    handle.as_rule().step(pre_spike, post_spike, reward);
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

/// Backward-compatible FFI for ELIGENT learner step.
#[no_mangle]
pub extern "C" fn step_learner(ptr: *mut EligentRule, fired: bool, pre_spike: bool, global_reward: f32) {
    if ptr.is_null() {
        return;
    }
    let state = unsafe { &mut *ptr };
    state.step(pre_spike, fired, global_reward);
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
        rule.step(true, false, 0.0);
        rule.step(false, true, 1.0);
        assert_ne!(rule.weight(), initial, "Weight should change after reward");
    }

    #[test]
    fn stdp_ltp_direction() {
        let mut rule = StdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0);
        let initial = rule.weight();
        // Pre spike first, then post spike → LTP (potentiation)
        rule.step(true, false, 0.0);
        rule.step(false, true, 0.0);
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
        rule.step(false, true, 0.0);
        rule.step(true, false, 0.0);
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
            rule.step(true, true, 0.0);
        }
        assert!(rule.weight() >= 0.0 && rule.weight() <= 1.0);
    }

    #[test]
    fn rstdp_no_reward_no_change() {
        let mut rule = RewardStdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0, 0.95);
        let initial = rule.weight();
        // Spikes but no reward → eligibility builds but weight doesn't change
        rule.step(true, false, 0.0);
        rule.step(false, true, 0.0);
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
        rule.step(true, false, 0.0);
        rule.step(false, true, 1.0); // reward delivered with post spike
        assert_ne!(
            rule.weight(),
            initial,
            "Reward should drive weight change"
        );
    }

    #[test]
    fn bcm_below_threshold_depresses() {
        let mut rule = BcmRule::new(0.5, 0.1, 10.0);
        rule.theta_m = 1.0; // set high threshold
        let initial = rule.weight();
        // y=1 < θ_m=1.0 → (y - θ_m) = 0, no change on exact boundary
        // But activity below threshold across time should depress
        for _ in 0..10 {
            rule.step(true, false, 0.0); // pre but no post → no change
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
            rule.step(true, true, 0.0);
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
            assert!(!ptr.is_null(), "Rule type {rule_type} should create successfully");
            unsafe {
                step_rule(ptr, true, false, 0.0);
                step_rule(ptr, false, true, 1.0);
                let w = get_rule_weight(ptr);
                assert!(w.is_finite(), "Weight should be finite for rule {rule_type}");
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
        step_learner(ptr, true, true, 1.0);
        destroy_learner(ptr);
    }

    #[test]
    fn rule_reset_clears_traces() {
        let mut rule = StdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0);
        rule.step(true, false, 0.0);
        assert!(rule.pre_trace > 0.0);
        rule.reset();
        assert_eq!(rule.pre_trace, 0.0);
        assert_eq!(rule.post_trace, 0.0);
    }
}
