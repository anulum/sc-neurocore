// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for mckean

package services

import (
	"math"
)

// McKeanNeuronState holds the neuron state
type McKeanNeuronState struct {
	V       float64
	W       float64
	A       float64
	Epsilon float64
	Gamma   float64
	Dt      float64
	VPeak   float64
}

// NewMcKeanNeuron creates a new McKeanNeuron neuron with default parameters
func NewMcKeanNeuron() *McKeanNeuronState {
	return &McKeanNeuronState{
		V:       0.0,
		W:       0.0,
		A:       0.25,
		Epsilon: 0.01,
		Gamma:   0.5,
		Dt:      0.1,
		VPeak:   0.8,
	}
}

// ValidateMcKeanNeuron checks that the piecewise-linear dynamics are finite and well-formed.
func ValidateMcKeanNeuron(s *McKeanNeuronState) bool {
	return s != nil &&
		!math.IsNaN(s.V) && !math.IsInf(s.V, 0) &&
		!math.IsNaN(s.W) && !math.IsInf(s.W, 0) &&
		!math.IsNaN(s.A) && !math.IsInf(s.A, 0) && s.A > 0 && s.A < 1 &&
		!math.IsNaN(s.Epsilon) && !math.IsInf(s.Epsilon, 0) && s.Epsilon > 0 &&
		!math.IsNaN(s.Gamma) && !math.IsInf(s.Gamma, 0) && s.Gamma > 0 &&
		!math.IsNaN(s.Dt) && !math.IsInf(s.Dt, 0) && s.Dt > 0 &&
		!math.IsNaN(s.VPeak) && !math.IsInf(s.VPeak, 0)
}

func (s *McKeanNeuronState) f(v float64) float64 {
	mid1 := s.A / 2.0
	mid2 := (1.0 + s.A) / 2.0
	if v < mid1 {
		return -v
	}
	if v < mid2 {
		return v - s.A
	}
	return 1.0 - v
}

// Step advances the neuron by one timestep
func (s *McKeanNeuronState) Step(iExt float64) int {
	if !ValidateMcKeanNeuron(s) || math.IsNaN(iExt) || math.IsInf(iExt, 0) {
		return 0
	}

	dv := (s.f(s.V) - s.W + iExt) * s.Dt
	dw := s.Epsilon * (s.V - s.Gamma*s.W) * s.Dt
	vPrev := s.V
	newV := s.V + dv
	newW := s.W + dw
	if math.IsNaN(newV) || math.IsInf(newV, 0) || math.IsNaN(newW) || math.IsInf(newW, 0) {
		return 0
	}
	s.V = newV
	s.W = newW
	if s.V >= s.VPeak && vPrev < s.VPeak {
		return 1
	}
	return 0
}

// SimulateMcKeanNeuron runs the neuron for n steps
func SimulateMcKeanNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMcKeanNeuron()
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
