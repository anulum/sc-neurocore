// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for escape_rate

package services

import (
	"math"
)

// EscapeRateNeuronState holds the neuron state
type EscapeRateNeuronState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	TauM float64
	Rho0 float64
	DeltaU float64
	Resistance float64
	Dt float64
}

// NewEscapeRateNeuron creates a new EscapeRateNeuron neuron with default parameters
func NewEscapeRateNeuron() *EscapeRateNeuronState {
	return &EscapeRateNeuronState{
		V: -70.0,
		VRest: -70.0,
		VReset: -70.0,
		VThreshold: -50.0,
		TauM: 10.0,
		Rho0: 0.001,
		DeltaU: 3.0,
		Resistance: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *EscapeRateNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateEscapeRateNeuron runs the neuron for n steps
func SimulateEscapeRateNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewEscapeRateNeuron()
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
