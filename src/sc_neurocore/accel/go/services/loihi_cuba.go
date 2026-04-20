// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for loihi_cuba

package services

import (
	"math"
)

// LoihiCUBANeuronState holds the neuron state
type LoihiCUBANeuronState struct {
	V float64
	U float64
	TauV float64
	TauU float64
	VThreshold float64
	VReset float64
}

// NewLoihiCUBANeuron creates a new LoihiCUBANeuron neuron with default parameters
func NewLoihiCUBANeuron() *LoihiCUBANeuronState {
	return &LoihiCUBANeuronState{
		V: 0.0,
		U: 0.0,
		TauV: 10.0,
		TauU: 5.0,
		VThreshold: 1000.0,
		VReset: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *LoihiCUBANeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateLoihiCUBANeuron runs the neuron for n steps
func SimulateLoihiCUBANeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewLoihiCUBANeuron()
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
