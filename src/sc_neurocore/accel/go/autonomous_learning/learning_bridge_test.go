// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Cgo bridge test for autonomous_learning Rust C-FFI

package autonomous_learning

import (
	"errors"
	"math"
	"testing"
)

func requirePanic(t *testing.T, operation func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected operation to panic")
		}
	}()
	operation()
}

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
	if err := rule.Step(true, false, 0.0); err != nil {
		t.Fatal(err)
	}
	if err := rule.StepDt(false, true, 0.0, DefaultDt); err != nil {
		t.Fatal(err)
	}

	newWeight := rule.Weight()
	if newWeight <= initialWeight {
		t.Errorf("Expected weight to increase after LTP timing, but got %f from %f", newWeight, initialWeight)
	}

	if err := rule.Reset(); err != nil {
		t.Fatal(err)
	}
}

func TestEligentLearner(t *testing.T) {
	learner := NewEligentLearner(1.0, 0.1, 0.5)
	if learner == nil {
		t.Skip("C-FFI Library not found, skipping integration test.")
		return
	}
	defer learner.Destroy()

	// Should not crash when stepping
	if err := learner.Step(true, true, 1.0); err != nil {
		t.Fatal(err)
	}
	if err := learner.StepDt(true, true, 0, 0); !errors.Is(err, ErrInvalidTimestep) {
		t.Fatalf("expected invalid timestep error, got %v", err)
	}
	learner.Destroy()
	if err := learner.Step(true, true, 0); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected closed handle error, got %v", err)
	}
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
	if err := layer.Step(pre, post, rewards); err != nil {
		t.Fatal(err)
	}
	if err := layer.StepDt(pre, post, rewards, DefaultDt); err != nil {
		t.Fatal(err)
	}

	weights, err := layer.TryGetWeights()
	if err != nil {
		t.Fatal(err)
	}
	if len(weights) != 10 {
		t.Errorf("Expected 10 weights, got %d", len(weights))
	}
}

func TestConstructorsRejectUnsafeDomains(t *testing.T) {
	nan := float32(math.NaN())
	if NewPlasticityRule(99, 0.5, 0.1, 0.05) != nil {
		t.Fatal("invalid rule type must fail")
	}
	if NewPlasticityRule(RuleStdp, nan, 0.1, 0.05) != nil {
		t.Fatal("non-finite rule weight must fail")
	}
	if NewPlasticityRule(RuleStdp, 0.5, -0.1, 0.05) != nil {
		t.Fatal("negative rule parameter must fail")
	}
	if NewEligentLearner(0, 0.1, 0.5) != nil {
		t.Fatal("non-positive threshold must fail")
	}
	if NewEligentLearner(1, -0.1, 0.5) != nil {
		t.Fatal("negative target rate must fail")
	}
	if NewRuleLayer(0, RuleStdp, 0.5, 0.1, 0.05) != nil {
		t.Fatal("zero-length layer must fail before taking slice addresses")
	}
	if NewRuleLayer(1, RuleStdp, 1.1, 0.1, 0.05) != nil {
		t.Fatal("out-of-domain layer weight must fail")
	}
}

func TestRuleRejectsClosedAndInvalidStep(t *testing.T) {
	rule := NewPlasticityRule(RuleStdp, 0.5, 0.1, 0.05)
	if rule == nil {
		t.Skip("C-FFI Library not found, skipping integration test.")
	}
	if err := rule.StepDt(true, false, 0, 0); !errors.Is(err, ErrInvalidTimestep) {
		t.Fatalf("expected invalid timestep error, got %v", err)
	}
	if err := rule.StepDt(true, false, float32(math.NaN()), DefaultDt); !errors.Is(err, ErrInvalidReward) {
		t.Fatalf("expected invalid reward error, got %v", err)
	}
	rule.Destroy()
	if _, err := rule.TryWeight(); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected closed handle error, got %v", err)
	}
	if err := rule.Reset(); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected closed handle error, got %v", err)
	}
	requirePanic(t, func() { rule.Weight() })
	var absent *PlasticityRule
	if err := absent.Step(true, false, 0); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected closed handle error, got %v", err)
	}
	absent.Destroy()
}

func TestLayerRejectsShortAndInvalidBuffers(t *testing.T) {
	layer := NewRuleLayer(2, RuleStdp, 0.5, 0.1, 0.05)
	if layer == nil {
		t.Skip("C-FFI Library not found, skipping integration test.")
	}
	defer layer.Destroy()
	if err := layer.Step([]bool{true}, []bool{false}, []float32{0}); !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("expected length mismatch error, got %v", err)
	}
	if err := layer.StepDt(
		[]bool{true, false}, []bool{false, true}, []float32{0, 0}, 0,
	); !errors.Is(err, ErrInvalidTimestep) {
		t.Fatalf("expected invalid timestep error, got %v", err)
	}
	if err := layer.Step(
		[]bool{true, false}, []bool{false, true}, []float32{0, float32(math.Inf(1))},
	); !errors.Is(err, ErrInvalidReward) {
		t.Fatalf("expected invalid reward error, got %v", err)
	}
	layer.Destroy()
	if _, err := layer.TryGetWeights(); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected closed handle error, got %v", err)
	}
	requirePanic(t, func() { layer.GetWeights() })
}
