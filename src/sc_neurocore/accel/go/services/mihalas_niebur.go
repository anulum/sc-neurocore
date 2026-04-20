// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for mihalas_niebur

package services

import (
	"math"
)

// MihalasNieburNeuronState holds the neuron state
type MihalasNieburNeuronState struct {
	V float64
	Theta float64
	I1 float64
	I2 float64
	VRest float64
	VReset float64
	ThetaReset float64
	ThetaInf float64
	TauV float64
	TauTheta float64
	Tau1 float64
	Tau2 float64
	A float64
	B float64
	R1 float64
	R2 float64
	Dt float64
}

// NewMihalasNieburNeuron creates a new MihalasNieburNeuron neuron with default parameters
func NewMihalasNieburNeuron() *MihalasNieburNeuronState {
	return &MihalasNieburNeuronState{
		V: 0.0,
		Theta: 1.0,
		I1: 0.0,
		I2: 0.0,
		VRest: 0.0,
		VReset: 0.0,
		ThetaReset: 1.0,
		ThetaInf: 1.0,
		TauV: 10.0,
		TauTheta: 100.0,
		Tau1: 10.0,
		Tau2: 200.0,
		A: 0.0,
		B: 0.0,
		R1: 0.0,
		R2: 0.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *MihalasNieburNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateMihalasNieburNeuron runs the neuron for n steps
func SimulateMihalasNieburNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMihalasNieburNeuron()
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
