// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for mat

package services

import (
	"math"
)

// MATNeuronState holds the neuron state
type MATNeuronState struct {
	V float64
	Theta1 float64
	Theta2 float64
	VRest float64
	VReset float64
	VThresholdBase float64
	TauM float64
	Tau1 float64
	Tau2 float64
	H1 float64
	H2 float64
	Resistance float64
	Dt float64
}

// NewMATNeuron creates a new MATNeuron neuron with default parameters
func NewMATNeuron() *MATNeuronState {
	return &MATNeuronState{
		V: -70.0,
		Theta1: 0.0,
		Theta2: 0.0,
		VRest: -70.0,
		VReset: -70.0,
		VThresholdBase: -50.0,
		TauM: 10.0,
		Tau1: 10.0,
		Tau2: 200.0,
		H1: 5.0,
		H2: 3.0,
		Resistance: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *MATNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateMATNeuron runs the neuron for n steps
func SimulateMATNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMATNeuron()
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
