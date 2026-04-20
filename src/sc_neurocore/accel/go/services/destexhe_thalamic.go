// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for destexhe_thalamic

package services

import (
	"math"
)

// DestexheThalamicNeuronState holds the neuron state
type DestexheThalamicNeuronState struct {
	V float64
	HNa float64
	NK float64
	MT float64
	HT float64
	GNa float64
	GK float64
	GT float64
	GL float64
	ENa float64
	EK float64
	ECa float64
	EL float64
	Dt float64
	VThreshold float64
}

// NewDestexheThalamicNeuron creates a new DestexheThalamicNeuron neuron with default parameters
func NewDestexheThalamicNeuron() *DestexheThalamicNeuronState {
	return &DestexheThalamicNeuronState{
		V: -65.0,
		HNa: 0.6,
		NK: 0.3,
		MT: 0.0,
		HT: 1.0,
		GNa: 100.0,
		GK: 10.0,
		GT: 2.0,
		GL: 0.05,
		ENa: 50.0,
		EK: -90.0,
		ECa: 120.0,
		EL: -70.0,
		Dt: 0.02,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *DestexheThalamicNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateDestexheThalamicNeuron runs the neuron for n steps
func SimulateDestexheThalamicNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDestexheThalamicNeuron()
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
