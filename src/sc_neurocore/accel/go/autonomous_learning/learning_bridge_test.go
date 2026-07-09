// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Cgo bridge test for autonomous_learning Rust C-FFI

package autonomous_learning

import (
	"testing"
)

func TestNewPlasticityRule(t *testing.T) {
	rule := NewPlasticityRule(RuleStdp, 0.5, 0.1, 0.05)
	if rule == nil {
		t.Skip("C-FFI Library not found, skipping integration test.")
		return
	}
	defer rule.Destroy()

	initialWeight := rule.Weight()
	if initialWeight != 0.5 {
		t.Errorf("Expected initial weight to be 0.5, got %f", initialWeight)
	}

	// STDP pre-before-post should potentiate
	rule.Step(true, false, 0.0)
	rule.StepDt(false, true, 0.0, DefaultDt)

	newWeight := rule.Weight()
	if newWeight <= initialWeight {
		t.Errorf("Expected weight to increase after LTP timing, but got %f from %f", newWeight, initialWeight)
	}

	rule.Reset()
}

func TestEligentLearner(t *testing.T) {
	learner := NewEligentLearner(1.0, 0.1, 0.5)
	if learner == nil {
		t.Skip("C-FFI Library not found, skipping integration test.")
		return
	}
	defer learner.Destroy()

	// Should not crash when stepping
	learner.Step(true, true, 1.0)
}

func TestRuleLayer(t *testing.T) {
	layer := NewRuleLayer(10, RuleStdp, 0.5, 0.1, 0.05)
	if layer == nil {
		t.Skip("C-FFI Library not found, skipping integration test.")
		return
	}
	defer layer.Destroy()

	pre := make([]bool, 10)
	post := make([]bool, 10)
	rewards := make([]float32, 10)

	// Simulate one step for the entire layer
	layer.Step(pre, post, rewards)
	layer.StepDt(pre, post, rewards, DefaultDt)

	weights := layer.GetWeights()
	if len(weights) != 10 {
		t.Errorf("Expected 10 weights, got %d", len(weights))
	}
}
