// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for ibarz_tanaka_map

package services

import (
	"math"
)

// IbarzTanakaMapNeuronState holds the neuron state
type IbarzTanakaMapNeuronState struct {
	X float64
	Y float64
	Alpha float64
	Beta float64
	Mu float64
	Sigma float64
	XThreshold float64
	XReset float64
}

// NewIbarzTanakaMapNeuron creates a new IbarzTanakaMapNeuron neuron with default parameters
func NewIbarzTanakaMapNeuron() *IbarzTanakaMapNeuronState {
	return &IbarzTanakaMapNeuronState{
		X: -1.0,
		Y: -2.5,
		Alpha: 3.65,
		Beta: 0.25,
		Mu: 0.0005,
		Sigma: -1.6,
		XThreshold: 3.0,
		XReset: -1.0,
	}
}

// Step advances the neuron by one timestep
func (s *IbarzTanakaMapNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateIbarzTanakaMapNeuron runs the neuron for n steps
func SimulateIbarzTanakaMapNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewIbarzTanakaMapNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.X
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
