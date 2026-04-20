// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for chandelier_neuron

package services

import (
	"math"
)

// ChandelierNeuronState holds the neuron state
type ChandelierNeuronState struct {
	V float64
	H float64
	N float64
	D float64
	P float64
	GNa float64
	GK float64
	GKv1 float64
	GKv3 float64
	GL float64
	ENa float64
	EK float64
	EL float64
	CM float64
	Phi float64
	Dt float64
	VThreshold float64
}

// NewChandelierNeuron creates a new ChandelierNeuron neuron with default parameters
func NewChandelierNeuron() *ChandelierNeuronState {
	return &ChandelierNeuronState{
		V: -65.0,
		H: 0.8,
		N: 0.1,
		D: 0.0,
		P: 0.0,
		GNa: 35.0,
		GK: 9.0,
		GKv1: 3.0,
		GKv3: 4.0,
		GL: 0.1,
		ENa: 55.0,
		EK: -90.0,
		EL: -65.0,
		CM: 1.0,
		Phi: 5.0,
		Dt: 0.01,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *ChandelierNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateChandelierNeuron runs the neuron for n steps
func SimulateChandelierNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewChandelierNeuron()
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
