// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Cgo bridge for autonomous_learning Rust C-FFI

package autonomous_learning

// #cgo LDFLAGS: -L../../../_native -lautonomous_learning
// #include <stdbool.h>
// #include <stdint.h>
//
// void* create_rule(uint32_t rule_type, float weight, float param_a, float param_b);
// void step_rule(void* ptr, bool pre_spike, bool post_spike, float reward, float dt);
// float get_rule_weight(void* ptr);
// void reset_rule(void* ptr);
// void destroy_rule(void* ptr);
//
// void* create_learner(float threshold, float target_rate, float weight);
// void step_learner(void* ptr, bool fired, bool pre_spike, float global_reward, float dt);
// void destroy_learner(void* ptr);
//
// void* create_rule_layer(size_t count, uint32_t rule_type, float weight, float param_a, float param_b);
// void step_rule_layer(void* layer_ptr, const bool* pre_spikes, const bool* post_spikes, const float* rewards, float dt);
// void get_rule_layer_weights(const void* layer_ptr, float* out_weights);
// void destroy_rule_layer(void* layer_ptr);
import "C"
import "unsafe"

// Rule Type Constants
const (
	RuleEligent    = 0
	RuleStdp       = 1
	RuleRewardStdp = 2
	RuleBcm        = 3
)

// DefaultDt matches the Python bridge default timestep for one-step helpers.
const DefaultDt float32 = 0.001

// PlasticityRule represents a handle to the Rust plasticity engine.
type PlasticityRule struct {
	ptr unsafe.Pointer
}

// NewPlasticityRule creates a new rule instance safely.
func NewPlasticityRule(ruleType uint32, weight, paramA, paramB float32) *PlasticityRule {
	ptr := C.create_rule(C.uint32_t(ruleType), C.float(weight), C.float(paramA), C.float(paramB))
	if ptr == nil {
		return nil
	}
	return &PlasticityRule{ptr: ptr}
}

// Step advances the rule by one timestep.
func (r *PlasticityRule) Step(preSpike, postSpike bool, reward float32) {
	r.StepDt(preSpike, postSpike, reward, DefaultDt)
}

// StepDt advances the rule by one timestep using an explicit timestep.
func (r *PlasticityRule) StepDt(preSpike, postSpike bool, reward, dt float32) {
	C.step_rule(r.ptr, C.bool(preSpike), C.bool(postSpike), C.float(reward), C.float(dt))
}

// Weight gets the current computed weight from the rule.
func (r *PlasticityRule) Weight() float32 {
	return float32(C.get_rule_weight(r.ptr))
}

// Reset clears traces / intermediate states.
func (r *PlasticityRule) Reset() {
	C.reset_rule(r.ptr)
}

// Destroy frees the memory on the Rust side.
func (r *PlasticityRule) Destroy() {
	if r.ptr != nil {
		C.destroy_rule(r.ptr)
		r.ptr = nil
	}
}

// EligentLearner represents a specific backward-compatible ELIGENT interface.
type EligentLearner struct {
	ptr unsafe.Pointer
}

// NewEligentLearner creates a new learner instance safely.
func NewEligentLearner(threshold, targetRate, weight float32) *EligentLearner {
	ptr := C.create_learner(C.float(threshold), C.float(targetRate), C.float(weight))
	if ptr == nil {
		return nil
	}
	return &EligentLearner{ptr: ptr}
}

// Step advances the learner.
func (l *EligentLearner) Step(fired, preSpike bool, globalReward float32) {
	l.StepDt(fired, preSpike, globalReward, DefaultDt)
}

// StepDt advances the learner using an explicit timestep.
func (l *EligentLearner) StepDt(fired, preSpike bool, globalReward, dt float32) {
	C.step_learner(l.ptr, C.bool(fired), C.bool(preSpike), C.float(globalReward), C.float(dt))
}

// Destroy frees the memory on the Rust side.
func (l *EligentLearner) Destroy() {
	if l.ptr != nil {
		C.destroy_learner(l.ptr)
		l.ptr = nil
	}
}

// RuleLayer represents a concurrent layer of plasticity rules.
type RuleLayer struct {
	ptr   unsafe.Pointer
	count int
}

// NewRuleLayer creates an array of rules natively.
func NewRuleLayer(count int, ruleType uint32, weight, paramA, paramB float32) *RuleLayer {
	ptr := C.create_rule_layer(C.size_t(count), C.uint32_t(ruleType), C.float(weight), C.float(paramA), C.float(paramB))
	if ptr == nil {
		return nil
	}
	return &RuleLayer{ptr: ptr, count: count}
}

// Step advances the entire layer concurrently.
func (l *RuleLayer) Step(preSpikes, postSpikes []bool, rewards []float32) {
	l.StepDt(preSpikes, postSpikes, rewards, DefaultDt)
}

// StepDt advances the entire layer concurrently using an explicit timestep.
func (l *RuleLayer) StepDt(preSpikes, postSpikes []bool, rewards []float32, dt float32) {
	if len(preSpikes) != l.count || len(postSpikes) != l.count || len(rewards) != l.count {
		panic("Slice size mismatch in RuleLayer.Step")
	}
	C.step_rule_layer(
		l.ptr,
		(*C.bool)(unsafe.Pointer(&preSpikes[0])),
		(*C.bool)(unsafe.Pointer(&postSpikes[0])),
		(*C.float)(unsafe.Pointer(&rewards[0])),
		C.float(dt),
	)
}

// GetWeights retrieves all concurrent weights in a single slice.
func (l *RuleLayer) GetWeights() []float32 {
	out := make([]float32, l.count)
	C.get_rule_layer_weights(
		l.ptr,
		(*C.float)(unsafe.Pointer(&out[0])),
	)
	return out
}

// Destroy cleans up the layer handles.
func (l *RuleLayer) Destroy() {
	if l.ptr != nil {
		C.destroy_rule_layer(l.ptr)
		l.ptr = nil
	}
}
