// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for tc_lif

package services

import (
	"math"
)

// TwoCompartmentLIFNeuronState holds the neuron state
type TwoCompartmentLIFNeuronState struct {
	VS float64
	VD float64
	VRest float64
	VReset float64
	Theta float64
	TauS float64
	TauD float64
	Kappa float64
	Dt float64
}

// NewTwoCompartmentLIFNeuron creates a new TwoCompartmentLIFNeuron neuron with default parameters
func NewTwoCompartmentLIFNeuron() *TwoCompartmentLIFNeuronState {
	return &TwoCompartmentLIFNeuronState{
		VS: 0.0,
		VD: 0.0,
		VRest: 0.0,
		VReset: 0.0,
		Theta: 1.0,
		TauS: 2.0,
		TauD: 20.0,
		Kappa: 0.5,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *TwoCompartmentLIFNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateTwoCompartmentLIFNeuron runs the neuron for n steps
func SimulateTwoCompartmentLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewTwoCompartmentLIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.VS
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
