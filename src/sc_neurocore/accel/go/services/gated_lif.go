// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for gated_lif

package services

import (
	"math"
)

// GatedLIFNeuronState holds the neuron state
type GatedLIFNeuronState struct {
	V float64
	GateV float64
	GateI float64
	VThreshold float64
	Dt float64
}

// NewGatedLIFNeuron creates a new GatedLIFNeuron neuron with default parameters
func NewGatedLIFNeuron() *GatedLIFNeuronState {
	return &GatedLIFNeuronState{
		V: 0.0,
		GateV: 0.9,
		GateI: 1.0,
		VThreshold: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *GatedLIFNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = 0.0
		return 1
	}
	return 0
}

// SimulateGatedLIFNeuron runs the neuron for n steps
func SimulateGatedLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGatedLIFNeuron()
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
