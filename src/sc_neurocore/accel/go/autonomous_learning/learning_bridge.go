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
import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// Rule Type Constants
const (
	RuleEligent    = 0
	RuleStdp       = 1
	RuleRewardStdp = 2
	RuleBcm        = 3
)

// DefaultDt matches the Python bridge default timestep for one-step helpers.
const DefaultDt float32 = 0.001

var (
	// ErrClosed reports use of a nil or already destroyed native handle.
	ErrClosed = errors.New("autonomous-learning handle is closed")
	// ErrInvalidTimestep reports a non-finite or non-positive timestep.
	ErrInvalidTimestep = errors.New("learning timestep must be finite and positive")
	// ErrInvalidReward reports a non-finite reward value.
	ErrInvalidReward = errors.New("learning reward must be finite")
	// ErrLengthMismatch reports buffers that do not match the layer count.
	ErrLengthMismatch = errors.New("learning-layer slice size mismatch")
)

func finite(value float32) bool {
	converted := float64(value)
	return !math.IsNaN(converted) && !math.IsInf(converted, 0)
}

func validRuleType(ruleType uint32) bool {
	return ruleType <= RuleBcm
}

func validWeight(weight float32) bool {
	return finite(weight) && weight >= 0 && weight <= 1
}

func validateTimestep(dt float32) error {
	if !finite(dt) || dt <= 0 {
		return fmt.Errorf("%w: got %v", ErrInvalidTimestep, dt)
	}
	return nil
}

func validateReward(reward float32) error {
	if !finite(reward) {
		return fmt.Errorf("%w: got %v", ErrInvalidReward, reward)
	}
	return nil
}

// PlasticityRule represents a handle to the Rust plasticity engine.
type PlasticityRule struct {
	ptr unsafe.Pointer
}

// NewPlasticityRule creates a new rule instance safely.
func NewPlasticityRule(ruleType uint32, weight, paramA, paramB float32) *PlasticityRule {
	if !validRuleType(ruleType) || !validWeight(weight) || !finite(paramA) || paramA < 0 || !finite(paramB) || paramB < 0 {
		return nil
	}
	ptr := C.create_rule(C.uint32_t(ruleType), C.float(weight), C.float(paramA), C.float(paramB))
	if ptr == nil {
		return nil
	}
	return &PlasticityRule{ptr: ptr}
}

// Step advances the rule by one timestep.
func (r *PlasticityRule) Step(preSpike, postSpike bool, reward float32) error {
	return r.StepDt(preSpike, postSpike, reward, DefaultDt)
}

// StepDt advances the rule by one timestep using an explicit timestep.
func (r *PlasticityRule) StepDt(preSpike, postSpike bool, reward, dt float32) error {
	if r == nil || r.ptr == nil {
		return ErrClosed
	}
	if err := validateReward(reward); err != nil {
		return err
	}
	if err := validateTimestep(dt); err != nil {
		return err
	}
	C.step_rule(r.ptr, C.bool(preSpike), C.bool(postSpike), C.float(reward), C.float(dt))
	return nil
}

// TryWeight gets the current computed weight or reports a closed handle.
func (r *PlasticityRule) TryWeight() (float32, error) {
	if r == nil || r.ptr == nil {
		return 0, ErrClosed
	}
	return float32(C.get_rule_weight(r.ptr)), nil
}

// Weight gets the current computed weight, panicking only for legacy callers
// that use a closed handle. New code should use TryWeight.
func (r *PlasticityRule) Weight() float32 {
	weight, err := r.TryWeight()
	if err != nil {
		panic(err)
	}
	return weight
}

// Reset clears traces / intermediate states.
func (r *PlasticityRule) Reset() error {
	if r == nil || r.ptr == nil {
		return ErrClosed
	}
	C.reset_rule(r.ptr)
	return nil
}

// Destroy frees the memory on the Rust side.
func (r *PlasticityRule) Destroy() {
	if r != nil && r.ptr != nil {
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
	if !finite(threshold) || threshold <= 0 || !finite(targetRate) || targetRate < 0 || !validWeight(weight) {
		return nil
	}
	ptr := C.create_learner(C.float(threshold), C.float(targetRate), C.float(weight))
	if ptr == nil {
		return nil
	}
	return &EligentLearner{ptr: ptr}
}

// Step advances the learner.
func (l *EligentLearner) Step(fired, preSpike bool, globalReward float32) error {
	return l.StepDt(fired, preSpike, globalReward, DefaultDt)
}

// StepDt advances the learner using an explicit timestep.
func (l *EligentLearner) StepDt(fired, preSpike bool, globalReward, dt float32) error {
	if l == nil || l.ptr == nil {
		return ErrClosed
	}
	if err := validateReward(globalReward); err != nil {
		return err
	}
	if err := validateTimestep(dt); err != nil {
		return err
	}
	C.step_learner(l.ptr, C.bool(fired), C.bool(preSpike), C.float(globalReward), C.float(dt))
	return nil
}

// Destroy frees the memory on the Rust side.
func (l *EligentLearner) Destroy() {
	if l != nil && l.ptr != nil {
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
	if count <= 0 || uint64(count) > uint64(^uint32(0)) || !validRuleType(ruleType) || !validWeight(weight) || !finite(paramA) || paramA < 0 || !finite(paramB) || paramB < 0 {
		return nil
	}
	ptr := C.create_rule_layer(C.size_t(count), C.uint32_t(ruleType), C.float(weight), C.float(paramA), C.float(paramB))
	if ptr == nil {
		return nil
	}
	return &RuleLayer{ptr: ptr, count: count}
}

// Step advances the entire layer concurrently.
func (l *RuleLayer) Step(preSpikes, postSpikes []bool, rewards []float32) error {
	return l.StepDt(preSpikes, postSpikes, rewards, DefaultDt)
}

// StepDt advances the entire layer concurrently using an explicit timestep.
func (l *RuleLayer) StepDt(preSpikes, postSpikes []bool, rewards []float32, dt float32) error {
	if l == nil || l.ptr == nil {
		return ErrClosed
	}
	if len(preSpikes) != l.count || len(postSpikes) != l.count || len(rewards) != l.count {
		return fmt.Errorf(
			"%w: expected %d, got pre=%d post=%d rewards=%d",
			ErrLengthMismatch,
			l.count,
			len(preSpikes),
			len(postSpikes),
			len(rewards),
		)
	}
	if err := validateTimestep(dt); err != nil {
		return err
	}
	for _, reward := range rewards {
		if err := validateReward(reward); err != nil {
			return err
		}
	}
	C.step_rule_layer(
		l.ptr,
		(*C.bool)(unsafe.Pointer(&preSpikes[0])),
		(*C.bool)(unsafe.Pointer(&postSpikes[0])),
		(*C.float)(unsafe.Pointer(&rewards[0])),
		C.float(dt),
	)
	return nil
}

// TryGetWeights retrieves all concurrent weights or reports a closed handle.
func (l *RuleLayer) TryGetWeights() ([]float32, error) {
	if l == nil || l.ptr == nil {
		return nil, ErrClosed
	}
	out := make([]float32, l.count)
	C.get_rule_layer_weights(
		l.ptr,
		(*C.float)(unsafe.Pointer(&out[0])),
	)
	return out, nil
}

// GetWeights retrieves all concurrent weights, panicking only for legacy
// callers that use a closed handle. New code should use TryGetWeights.
func (l *RuleLayer) GetWeights() []float32 {
	weights, err := l.TryGetWeights()
	if err != nil {
		panic(err)
	}
	return weights
}

// Destroy cleans up the layer handles.
func (l *RuleLayer) Destroy() {
	if l != nil && l.ptr != nil {
		C.destroy_rule_layer(l.ptr)
		l.ptr = nil
	}
}
