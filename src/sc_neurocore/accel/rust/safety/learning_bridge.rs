// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for learning_bridge

#![allow(unused_variables, dead_code, non_snake_case)]

use core::ffi::c_void;

extern "C" {
    fn create_rule(rule_type: u32, weight: f32, param_a: f32, param_b: f32) -> *mut c_void;
    fn step_rule(ptr: *mut c_void, pre_spike: bool, post_spike: bool, reward: f32);
    fn get_rule_weight(ptr: *const c_void) -> f32;
    fn reset_rule(ptr: *mut c_void);
    fn destroy_rule(ptr: *mut c_void);
    
    fn create_learner(threshold: f32, target_rate: f32, weight: f32) -> *mut c_void;
    fn step_learner(ptr: *mut c_void, fired: bool, pre_spike: bool, global_reward: f32);
    fn destroy_learner(ptr: *mut c_void);
}

pub struct RustPlasticityRule {
    ptr: *mut c_void,
}

impl RustPlasticityRule {
    pub fn new(rule_type: u32, weight: f32, param_a: f32, param_b: f32) -> Self {
        let ptr = unsafe { create_rule(rule_type, weight, param_a, param_b) };
        assert!(!ptr.is_null(), "Failed to create plasticity rule via FFI");
        Self { ptr }
    }

    pub fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f32) {
        unsafe { step_rule(self.ptr, pre_spike, post_spike, reward) }
    }

    pub fn weight(&self) -> f32 {
        unsafe { get_rule_weight(self.ptr) }
    }

    pub fn reset(&mut self) {
        unsafe { reset_rule(self.ptr) }
    }
}

impl Drop for RustPlasticityRule {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { destroy_rule(self.ptr) }
            self.ptr = std::ptr::null_mut();
        }
    }
}

#[derive(Debug)]
pub struct RustEligentLearner {
    ptr: *mut c_void,
}

impl RustEligentLearner {
    pub fn new(threshold: f32, target_rate: f32, weight: f32) -> Self {
        let ptr = unsafe { create_learner(threshold, target_rate, weight) };
        assert!(!ptr.is_null(), "Failed to create ELIGENT learner via FFI");
        Self { ptr }
    }

    pub fn step(&mut self, fired: bool, pre_spike: bool, global_reward: f32) {
        unsafe { step_learner(self.ptr, fired, pre_spike, global_reward) }
    }
}

impl Drop for RustEligentLearner {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { destroy_learner(self.ptr) }
            self.ptr = std::ptr::null_mut();
        }
    }
}

pub fn validate_learning_bridge(state: &RustEligentLearner) -> bool {
    !state.ptr.is_null()
}

// Tests require proper linking against libautonomous_learning.so
// which is managed externally during Python module build.

