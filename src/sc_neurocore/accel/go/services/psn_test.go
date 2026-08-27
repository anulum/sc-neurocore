// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for the sliding PSN

package services

import (
	"math"
	"testing"
)

func TestPSNMatchesPaperEquationOracle(t *testing.T) {
	drive := make([]float64, 32)
	for i := range drive {
		drive[i] = 0.4 + 0.3*math.Sin(float64(i)*0.17)
	}
	weights := []float64{0.1, -0.2, 0.35, 0.75}
	s := NewParallelSpikingNeuronWithKernel(4, 0.4)
	copy(s.Weights, weights)
	for step, current := range drive {
		spike, err := s.TryStep(current)
		if err != nil {
			t.Fatalf("finite configured drive rejected: %v", err)
		}
		hidden := 0.0
		for i, w := range weights {
			j := step - 3 + i
			x := 0.0
			if j >= 0 {
				x = drive[j]
			}
			hidden += w * x
		}
		if math.Float64bits(s.Hidden) != math.Float64bits(hidden) {
			t.Fatalf("step %d: hidden %v differs from oracle %v", step, s.Hidden, hidden)
		}
		want := 0
		if hidden >= 0.4 {
			want = 1
		}
		if spike != want {
			t.Fatalf("step %d: spike %d differs from oracle %d", step, spike, want)
		}
	}
}

func TestPSNFiringNeverClearsHistoryAndThetaRightContinuous(t *testing.T) {
	s := NewParallelSpikingNeuronWithKernel(1, 1.0)
	if s.Step(1.0) != 1 {
		t.Fatal("Theta(0) must fire at exact threshold")
	}
	if s.History[0] != 1.0 {
		t.Fatal("firing must not clear the retained inputs")
	}
}

func TestPSNInvalidInputRejectedAtomically(t *testing.T) {
	s := NewParallelSpikingNeuron()
	if _, err := s.TryStep(0.7); err != nil {
		t.Fatalf("finite drive rejected: %v", err)
	}
	hiddenBefore := s.Hidden
	historyBefore := append([]float64(nil), s.History...)
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := s.TryStep(bad); err == nil {
			t.Fatal("non-finite drive must be rejected")
		}
		if s.Step(bad) != 0 {
			t.Fatal("fail-closed Step must return 0 on rejected input")
		}
	}
	if s.Hidden != hiddenBefore {
		t.Fatal("rejected input mutated the hidden state")
	}
	for i, x := range s.History {
		if x != historyBefore[i] {
			t.Fatal("rejected input mutated the retained inputs")
		}
	}
}

func TestPSNResetClearsHistoryOnly(t *testing.T) {
	s := NewParallelSpikingNeuron()
	s.Step(1.0)
	s.Reset()
	for _, x := range s.History {
		if x != 0.0 {
			t.Fatal("reset must clear the retained inputs")
		}
	}
	if s.Hidden != 0.0 {
		t.Fatal("reset must clear the hidden state")
	}
	for _, w := range s.Weights {
		if w != 0.125 {
			t.Fatal("reset must preserve the weights")
		}
	}
}

func TestSimulateParallelSpikingNeuron(t *testing.T) {
	trace, spikes := SimulateParallelSpikingNeuron(20, 2.0)
	if len(trace) != 20 {
		t.Fatalf("trace length %d, want 20", len(trace))
	}
	if spikes == 0 {
		t.Fatal("constant supra-threshold drive must fire")
	}
}
