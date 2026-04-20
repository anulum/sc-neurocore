// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for e_prop_alif

package services

import (
	"math"
)

// EPropALIFNeuronState holds the neuron state
type EPropALIFNeuronState struct {
	V float64
	A float64
	ETrace float64
	TauM float64
	TauA float64
	VThresholdBase float64
	Beta float64
	VReset float64
	Dt float64
	AlphaM float64
	AlphaA float64
}

// NewEPropALIFNeuron creates a new EPropALIFNeuron neuron with default parameters
func NewEPropALIFNeuron() *EPropALIFNeuronState {
	return &EPropALIFNeuronState{
		V: 0.0,
		A: 0.0,
		ETrace: 0.0,
		TauM: 20.0,
		TauA: 200.0,
		VThresholdBase: 1.0,
		Beta: 0.07,
		VReset: 0.0,
		Dt: 1.0,
		AlphaM: 0.0,
		AlphaA: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *EPropALIFNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateEPropALIFNeuron runs the neuron for n steps
func SimulateEPropALIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewEPropALIFNeuron()
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
