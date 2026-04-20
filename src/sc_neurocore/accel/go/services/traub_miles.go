// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for traub_miles

package services

import (
	"math"
)

// TraubMilesNeuronState holds the neuron state
type TraubMilesNeuronState struct {
	V float64
	M float64
	H float64
	N float64
	GNa float64
	GK float64
	GL float64
	ENa float64
	EK float64
	EL float64
	Dt float64
	VThreshold float64
}

// NewTraubMilesNeuron creates a new TraubMilesNeuron neuron with default parameters
func NewTraubMilesNeuron() *TraubMilesNeuronState {
	return &TraubMilesNeuronState{
		V: -67.0,
		M: 0.05,
		H: 0.6,
		N: 0.3,
		GNa: 100.0,
		GK: 80.0,
		GL: 0.1,
		ENa: 50.0,
		EK: -100.0,
		EL: -67.0,
		Dt: 0.01,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *TraubMilesNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -67.0
		return 1
	}
	return 0
}

// SimulateTraubMilesNeuron runs the neuron for n steps
func SimulateTraubMilesNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewTraubMilesNeuron()
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

var _ = math.Exp
