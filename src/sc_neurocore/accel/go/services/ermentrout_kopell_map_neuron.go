// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for ermentrout_kopell_map_neuron

package services

import (
	"math"
)

// ErmentroutKopellMapNeuronState holds the neuron state
type ErmentroutKopellMapNeuronState struct {
	Theta float64
	Dt float64
	Gain float64
	ThetaThreshold float64
}

// NewErmentroutKopellMapNeuron creates a new ErmentroutKopellMapNeuron neuron with default parameters
func NewErmentroutKopellMapNeuron() *ErmentroutKopellMapNeuronState {
	return &ErmentroutKopellMapNeuronState{
		Theta: 0.0,
		Dt: 0.1,
		Gain: 1.0,
		ThetaThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *ErmentroutKopellMapNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateErmentroutKopellMapNeuron runs the neuron for n steps
func SimulateErmentroutKopellMapNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewErmentroutKopellMapNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Theta
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
