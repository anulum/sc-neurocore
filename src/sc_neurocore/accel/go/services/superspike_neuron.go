// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for superspike_neuron

package services

import (
	"math"
)

// SuperSpikeNeuronState holds the neuron state
type SuperSpikeNeuronState struct {
	V float64
	Trace float64
	TauM float64
	TauE float64
	VThreshold float64
	VReset float64
	BetaSg float64
	Dt float64
	AlphaM float64
	AlphaE float64
}

// NewSuperSpikeNeuron creates a new SuperSpikeNeuron neuron with default parameters
func NewSuperSpikeNeuron() *SuperSpikeNeuronState {
	return &SuperSpikeNeuronState{
		V: 0.0,
		Trace: 0.0,
		TauM: 10.0,
		TauE: 10.0,
		VThreshold: 1.0,
		VReset: 0.0,
		BetaSg: 10.0,
		Dt: 1.0,
		AlphaM: 0.0,
		AlphaE: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *SuperSpikeNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateSuperSpikeNeuron runs the neuron for n steps
func SimulateSuperSpikeNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSuperSpikeNeuron()
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
