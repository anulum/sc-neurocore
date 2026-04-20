// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for klif

package services

import (
	"math"
)

// KLIFNeuronState holds the neuron state
type KLIFNeuronState struct {
	V float64
	K float64
	Tau float64
	VThreshold float64
	VReset float64
	Dt float64
	Alpha float64
}

// NewKLIFNeuron creates a new KLIFNeuron neuron with default parameters
func NewKLIFNeuron() *KLIFNeuronState {
	return &KLIFNeuronState{
		V: 0.0,
		K: 1.0,
		Tau: 10.0,
		VThreshold: 1.0,
		VReset: 0.0,
		Dt: 1.0,
		Alpha: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *KLIFNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateKLIFNeuron runs the neuron for n steps
func SimulateKLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewKLIFNeuron()
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
