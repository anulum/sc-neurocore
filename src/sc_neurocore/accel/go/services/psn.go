// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for the sliding PSN

package services

import (
	"errors"
	"math"
)

// ParallelSpikingNeuronState holds the k-order sliding PSN
// (Fang et al. 2023): H[t] = sum_{i=0}^{k-1} W_i * X[t-k+1+i] with
// zero pre-history, S[t] = Theta(H[t] - VThreshold), Theta(0) = 1,
// no reset on firing. The sum accumulates sequentially from i = 0 so
// the result is bit-for-bit identical to every other backend.
type ParallelSpikingNeuronState struct {
	Weights    []float64
	History    []float64
	VThreshold float64
	Hidden     float64
}

// NewParallelSpikingNeuron creates a sliding PSN with the uniform
// repository default weights 1/k, k = 8, threshold 1.0.
func NewParallelSpikingNeuron() *ParallelSpikingNeuronState {
	return NewParallelSpikingNeuronWithKernel(8, 1.0)
}

// NewParallelSpikingNeuronWithKernel creates a sliding PSN with the
// uniform weights 1/kernelSize and the given threshold.
func NewParallelSpikingNeuronWithKernel(kernelSize int, vThreshold float64) *ParallelSpikingNeuronState {
	if kernelSize < 1 {
		kernelSize = 1
	}
	weights := make([]float64, kernelSize)
	for i := range weights {
		weights[i] = 1.0 / float64(kernelSize)
	}
	return &ParallelSpikingNeuronState{
		Weights:    weights,
		History:    make([]float64, kernelSize),
		VThreshold: vThreshold,
		Hidden:     0.0,
	}
}

func (s *ParallelSpikingNeuronState) valid() bool {
	if len(s.Weights) == 0 || len(s.History) != len(s.Weights) {
		return false
	}
	for _, w := range s.Weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			return false
		}
	}
	for _, x := range s.History {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return !math.IsNaN(s.VThreshold) && !math.IsInf(s.VThreshold, 0)
}

// TryStep advances one step; an invalid input, configuration, or
// non-finite hidden state returns an error with the state preserved.
func (s *ParallelSpikingNeuronState) TryStep(current float64) (int, error) {
	if math.IsNaN(current) || math.IsInf(current, 0) {
		return 0, errors.New("current must be finite")
	}
	if !s.valid() {
		return 0, errors.New("sliding PSN state and parameters must be finite")
	}

	hidden := 0.0
	for i, w := range s.Weights {
		value := current
		if i+1 < len(s.History) {
			value = s.History[i+1]
		}
		hidden += w * value
	}
	if math.IsNaN(hidden) || math.IsInf(hidden, 0) {
		return 0, errors.New("sliding PSN hidden state became non-finite")
	}

	copy(s.History, s.History[1:])
	s.History[len(s.History)-1] = current
	s.Hidden = hidden
	if hidden >= s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// Step is the fail-closed wrapper: returns 0 on any rejected input
// without mutating state.
func (s *ParallelSpikingNeuronState) Step(current float64) int {
	spike, err := s.TryStep(current)
	if err != nil {
		return 0
	}
	return spike
}

// Reset clears the retained inputs, preserving weights and threshold.
func (s *ParallelSpikingNeuronState) Reset() {
	for i := range s.History {
		s.History[i] = 0.0
	}
	s.Hidden = 0.0
}

// SimulateParallelSpikingNeuron runs the neuron for n steps under a
// constant drive and returns the hidden-state trace and spike count.
func SimulateParallelSpikingNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewParallelSpikingNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Hidden
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
