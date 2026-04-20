// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for martinotti_neuron

package services

import (
	"math"
)

// MartinottiNeuronState holds the neuron state
type MartinottiNeuronState struct {
	V float64
	M float64
	H float64
	N float64
	P float64
	S float64
	GNa float64
	GK float64
	GM float64
	GT float64
	GL float64
	ENa float64
	EK float64
	ECa float64
	EL float64
	CM float64
	Dt float64
	VThreshold float64
}

// NewMartinottiNeuron creates a new MartinottiNeuron neuron with default parameters
func NewMartinottiNeuron() *MartinottiNeuronState {
	return &MartinottiNeuronState{
		V: -65.0,
		M: 0.02,
		H: 0.8,
		N: 0.2,
		P: 0.0,
		S: 0.9,
		GNa: 40.0,
		GK: 5.0,
		GM: 0.25,
		GT: 0.01,
		GL: 0.05,
		ENa: 50.0,
		EK: -90.0,
		ECa: 120.0,
		EL: -65.0,
		CM: 0.8,
		Dt: 0.025,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *MartinottiNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateMartinottiNeuron runs the neuron for n steps
func SimulateMartinottiNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMartinottiNeuron()
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
