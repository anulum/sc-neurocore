// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for plif

package services

import (
	"math"
)

// ParametricLIFNeuronState holds the neuron state
type ParametricLIFNeuronState struct {
	V         float64
	A         float64
	Threshold float64
	Dt        float64
}

// NewParametricLIFNeuron creates a new ParametricLIFNeuron neuron with default parameters
func NewParametricLIFNeuron() *ParametricLIFNeuronState {
	return &ParametricLIFNeuronState{
		V:         0.0,
		A:         0.0,
		Threshold: 1.0,
		Dt:        1.0,
	}
}

// Step advances the neuron by one timestep
func (s *ParametricLIFNeuronState) Step(iExt float64) int {
	if !s.Valid() || !isFinitePLIF(iExt) {
		return 0
	}

	spike := 0.0
	if s.V >= s.Threshold {
		spike = 1.0
	}
	nextV := s.Alpha()*s.V*(1.0-spike) + iExt
	if !isFinitePLIF(nextV) {
		return 0
	}
	s.V = nextV
	if nextV >= s.Threshold {
		return 1
	}
	return 0
}

// Alpha returns the stable sigmoid of the learnable decay parameter.
func (s *ParametricLIFNeuronState) Alpha() float64 {
	if s.A >= 0.0 {
		z := math.Exp(-s.A)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(s.A)
	return z / (1.0 + z)
}

// Valid returns true when the state satisfies the PLIF physics contract.
func (s *ParametricLIFNeuronState) Valid() bool {
	return isFinitePLIF(s.V) &&
		isFinitePLIF(s.A) &&
		isFinitePLIF(s.Threshold) &&
		s.Threshold > 0.0 &&
		isFinitePLIF(s.Dt) &&
		s.Dt > 0.0
}

func (s *ParametricLIFNeuronState) Reset() {
	s.V = 0.0
}

func isFinitePLIF(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// SimulateParametricLIFNeuron runs the neuron for n steps
func SimulateParametricLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewParametricLIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
