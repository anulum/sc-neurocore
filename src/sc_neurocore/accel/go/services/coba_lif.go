// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for coba_lif

package services

import (
	"math"
)

// COBALIFNeuronState holds the neuron state
type COBALIFNeuronState struct {
	V float64
	GE float64
	GI float64
	CM float64
	GL float64
	EL float64
	EE float64
	EI float64
	TauE float64
	TauI float64
	VThreshold float64
	VReset float64
	Dt float64
}

// NewCOBALIFNeuron creates a new COBALIFNeuron neuron with default parameters
func NewCOBALIFNeuron() *COBALIFNeuronState {
	return &COBALIFNeuronState{
		V: -65.0,
		GE: 0.0,
		GI: 0.0,
		CM: 200.0,
		GL: 10.0,
		EL: -65.0,
		EE: 0.0,
		EI: -80.0,
		TauE: 5.0,
		TauI: 10.0,
		VThreshold: -50.0,
		VReset: -65.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *COBALIFNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateCOBALIFNeuron runs the neuron for n steps
func SimulateCOBALIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewCOBALIFNeuron()
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
