// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for sst_neuron

package services

import (
	"math"
)

// SSTNeuronState holds the neuron state
type SSTNeuronState struct {
	V float64
	M float64
	H float64
	N float64
	P float64
	S float64
	R float64
	GNa float64
	GK float64
	GM float64
	GT float64
	GH float64
	GL float64
	ENa float64
	EK float64
	ECa float64
	EH float64
	EL float64
	CM float64
	Dt float64
	VThreshold float64
}

// NewSSTNeuron creates a new SSTNeuron neuron with default parameters
func NewSSTNeuron() *SSTNeuronState {
	return &SSTNeuronState{
		V: -65.0,
		M: 0.02,
		H: 0.8,
		N: 0.2,
		P: 0.0,
		S: 0.9,
		R: 0.1,
		GNa: 50.0,
		GK: 5.0,
		GM: 0.12,
		GT: 0.01,
		GH: 0.02,
		GL: 0.05,
		ENa: 50.0,
		EK: -90.0,
		ECa: 120.0,
		EH: -40.0,
		EL: -65.0,
		CM: 1.0,
		Dt: 0.025,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *SSTNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateSSTNeuron runs the neuron for n steps
func SimulateSSTNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSSTNeuron()
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
