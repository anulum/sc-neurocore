// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for morris_lecar

package services

import (
	"math"
)

// MorrisLecarNeuronState holds the neuron state
type MorrisLecarNeuronState struct {
	V float64
	W float64
	CM float64
	GCa float64
	GK float64
	GL float64
	ECa float64
	EK float64
	EL float64
	V1 float64
	V2 float64
	V3 float64
	V4 float64
	Phi float64
	Dt float64
	VThreshold float64
}

// NewMorrisLecarNeuron creates a new MorrisLecarNeuron neuron with default parameters
func NewMorrisLecarNeuron() *MorrisLecarNeuronState {
	return &MorrisLecarNeuronState{
		V: -60.0,
		W: 0.0,
		CM: 20.0,
		GCa: 4.0,
		GK: 8.0,
		GL: 2.0,
		ECa: 120.0,
		EK: -84.0,
		EL: -60.0,
		V1: -1.2,
		V2: 18.0,
		V3: 12.0,
		V4: 17.4,
		Phi: 0.0,
		Dt: 0.1,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *MorrisLecarNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -60.0
		return 1
	}
	return 0
}

// SimulateMorrisLecarNeuron runs the neuron for n steps
func SimulateMorrisLecarNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMorrisLecarNeuron()
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
