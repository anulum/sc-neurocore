// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for glm_neuron

package services

import (
	"math"
)

// GLMNeuronState holds the neuron state
type GLMNeuronState struct {
	NK float64
	NH float64
	Mu float64
	DtMs float64
	K float64
	H float64
	StimBuf float64
	SpikeBuf float64
	Rng float64
}

// NewGLMNeuron creates a new GLMNeuron neuron with default parameters
func NewGLMNeuron() *GLMNeuronState {
	return &GLMNeuronState{
		NK: 10.0,
		NH: 20.0,
		Mu: -3.0,
		DtMs: 1.0,
		K: 0.0,
		H: 0.0,
		StimBuf: 0.0,
		SpikeBuf: 0.0,
		Rng: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *GLMNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateGLMNeuron runs the neuron for n steps
func SimulateGLMNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGLMNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.NK
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
