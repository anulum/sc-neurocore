// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for connor_stevens

package services

import (
	"math"
)

// ConnorStevensNeuronState holds the neuron state
type ConnorStevensNeuronState struct {
	V float64
	M float64
	H float64
	N float64
	A float64
	B float64
	GNa float64
	GK float64
	GA float64
	GL float64
	ENa float64
	EK float64
	EA float64
	EL float64
	CM float64
	Dt float64
	VThreshold float64
}

// NewConnorStevensNeuron creates a new ConnorStevensNeuron neuron with default parameters
func NewConnorStevensNeuron() *ConnorStevensNeuronState {
	return &ConnorStevensNeuronState{
		V: -68.0,
		M: 0.01,
		H: 0.99,
		N: 0.1,
		A: 0.5,
		B: 0.1,
		GNa: 120.0,
		GK: 20.0,
		GA: 47.7,
		GL: 0.3,
		ENa: 55.0,
		EK: -72.0,
		EA: -75.0,
		EL: -17.0,
		CM: 1.0,
		Dt: 0.01,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *ConnorStevensNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -68.0
		return 1
	}
	return 0
}

// SimulateConnorStevensNeuron runs the neuron for n steps
func SimulateConnorStevensNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewConnorStevensNeuron()
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
