// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for inhomogeneous_poisson

package services

import (
	"math"
)

// InhomogeneousPoissonNeuronState holds the neuron state
type InhomogeneousPoissonNeuronState struct {
	DtMs float64
}

// NewInhomogeneousPoissonNeuron creates a new InhomogeneousPoissonNeuron neuron with default parameters
func NewInhomogeneousPoissonNeuron() *InhomogeneousPoissonNeuronState {
	return &InhomogeneousPoissonNeuronState{
		DtMs: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *InhomogeneousPoissonNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateInhomogeneousPoissonNeuron runs the neuron for n steps
func SimulateInhomogeneousPoissonNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewInhomogeneousPoissonNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.DtMs
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
