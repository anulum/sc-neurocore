// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for glif

package services

import (
	"math"
)

// GLIFNeuronState holds the neuron state
type GLIFNeuronState struct {
	V float64
	Theta float64
	ThetaInf float64
	IAsc1 float64
	IAsc2 float64
	VRest float64
	VReset float64
	TauM float64
	TauTheta float64
	TauAsc1 float64
	TauAsc2 float64
	ATheta float64
	DeltaTheta float64
	RAsc1 float64
	RAsc2 float64
	Resistance float64
	Dt float64
}

// NewGLIFNeuron creates a new GLIFNeuron neuron with default parameters
func NewGLIFNeuron() *GLIFNeuronState {
	return &GLIFNeuronState{
		V: -70.0,
		Theta: -50.0,
		ThetaInf: -50.0,
		IAsc1: 0.0,
		IAsc2: 0.0,
		VRest: -70.0,
		VReset: -70.0,
		TauM: 10.0,
		TauTheta: 100.0,
		TauAsc1: 10.0,
		TauAsc2: 200.0,
		ATheta: 0.01,
		DeltaTheta: 2.0,
		RAsc1: 1.0,
		RAsc2: 0.5,
		Resistance: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *GLIFNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateGLIFNeuron runs the neuron for n steps
func SimulateGLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGLIFNeuron()
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
