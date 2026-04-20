// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for perfect_integrator

package services

import (
	"math"
)

// PerfectIntegratorNeuronState holds the neuron state
type PerfectIntegratorNeuronState struct {
	V float64
	CM float64
	VThreshold float64
	VReset float64
	Dt float64
}

// NewPerfectIntegratorNeuron creates a new PerfectIntegratorNeuron neuron with default parameters
func NewPerfectIntegratorNeuron() *PerfectIntegratorNeuronState {
	return &PerfectIntegratorNeuronState{
		V: 0.0,
		CM: 1.0,
		VThreshold: 1.0,
		VReset: 0.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *PerfectIntegratorNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulatePerfectIntegratorNeuron runs the neuron for n steps
func SimulatePerfectIntegratorNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPerfectIntegratorNeuron()
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
