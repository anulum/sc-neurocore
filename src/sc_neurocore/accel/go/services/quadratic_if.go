// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for quadratic_if

package services

import "math"

// QuadraticIFNeuronState holds the neuron state.
type QuadraticIFNeuronState struct {
	V      float64
	VReset float64
	VPeak  float64
	Dt     float64
}

// NewQuadraticIFNeuron creates a new QuadraticIFNeuron neuron with default parameters.
func NewQuadraticIFNeuron() *QuadraticIFNeuronState {
	return &QuadraticIFNeuronState{
		V:      -1.0,
		VReset: -1.0,
		VPeak:  1.0,
		Dt:     0.01,
	}
}

func quadraticIFFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// Valid reports whether the state satisfies the QIF integration contract.
func (s QuadraticIFNeuronState) Valid() bool {
	return quadraticIFFinite(s.V, s.VReset, s.VPeak, s.Dt) &&
		s.V < s.VPeak &&
		s.VReset < s.VPeak &&
		s.Dt > 0.0
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *QuadraticIFNeuronState) Step(iExt float64) int {
	if !quadraticIFFinite(iExt) || !s.Valid() {
		return 0
	}

	increment := (s.V*s.V + iExt) * s.Dt
	nextV := s.V + increment
	if !quadraticIFFinite(increment, nextV) {
		return 0
	}

	s.V = nextV
	if s.V >= s.VPeak {
		s.V = s.VReset
		return 1
	}
	return 0
}

// Reset restores dynamic state without changing parameters.
func (s *QuadraticIFNeuronState) Reset() {
	s.V = s.VReset
}

// SimulateQuadraticIFNeuron runs the neuron for n steps.
func SimulateQuadraticIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewQuadraticIFNeuron()
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
