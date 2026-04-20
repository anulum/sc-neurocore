// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for dcn_neuron

package services

import (
	"math"
)

// DCNNeuronState holds the neuron state
type DCNNeuronState struct {
	V float64
	H float64
	N float64
	P float64
	S float64
	R float64
	Ca float64
	GNa float64
	GNap float64
	GK float64
	GT float64
	GAhp float64
	GH float64
	GL float64
	ENa float64
	EK float64
	ECa float64
	EH float64
	EL float64
	CM float64
	VThreshold float64
}

// NewDCNNeuron creates a new DCNNeuron neuron with default parameters
func NewDCNNeuron() *DCNNeuronState {
	return &DCNNeuronState{
		V: -60.0,
		H: 0.6,
		N: 0.32,
		P: 0.01,
		S: 0.8,
		R: 0.1,
		Ca: 0.05,
		GNa: 35.0,
		GNap: 0.5,
		GK: 9.0,
		GT: 0.1,
		GAhp: 2.0,
		GH: 0.02,
		GL: 0.2,
		ENa: 55.0,
		EK: -90.0,
		ECa: 120.0,
		EH: -40.0,
		EL: -65.0,
		CM: 1.0,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *DCNNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -60.0
		return 1
	}
	return 0
}

// SimulateDCNNeuron runs the neuron for n steps
func SimulateDCNNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDCNNeuron()
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
