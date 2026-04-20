// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for pospischil

package services

import (
	"math"
)

// PospischilNeuronState holds the neuron state
type PospischilNeuronState struct {
	V float64
	M float64
	H float64
	N float64
	P float64
	GNa float64
	GKd float64
	GM float64
	GL float64
	ENa float64
	EK float64
	EL float64
	CM float64
	Vt float64
	Dt float64
	VThreshold float64
}

// NewPospischilNeuron creates a new PospischilNeuron neuron with default parameters
func NewPospischilNeuron() *PospischilNeuronState {
	return &PospischilNeuronState{
		V: -70.0,
		M: 0.05,
		H: 0.6,
		N: 0.3,
		P: 0.0,
		GNa: 50.0,
		GKd: 5.0,
		GM: 0.07,
		GL: 0.1,
		ENa: 50.0,
		EK: -90.0,
		EL: -70.0,
		CM: 1.0,
		Vt: -56.2,
		Dt: 0.025,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *PospischilNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -70.0
		return 1
	}
	return 0
}

// SimulatePospischilNeuron runs the neuron for n steps
func SimulatePospischilNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPospischilNeuron()
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
