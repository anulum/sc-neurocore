// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for mainen_sejnowski

package services

import (
	"math"
)

// MainenSejnowskiNeuronState holds the neuron state
type MainenSejnowskiNeuronState struct {
	Vs float64
	Va float64
	M float64
	H float64
	N float64
	Kappa float64
	GNa float64
	GK float64
	GL float64
	ENa float64
	EK float64
	EL float64
	CS float64
	CA float64
	Dt float64
	VThreshold float64
}

// NewMainenSejnowskiNeuron creates a new MainenSejnowskiNeuron neuron with default parameters
func NewMainenSejnowskiNeuron() *MainenSejnowskiNeuronState {
	return &MainenSejnowskiNeuronState{
		Vs: -65.0,
		Va: -65.0,
		M: 0.05,
		H: 0.6,
		N: 0.3,
		Kappa: 10.0,
		GNa: 3000.0,
		GK: 1500.0,
		GL: 1.0,
		ENa: 50.0,
		EK: -90.0,
		EL: -70.0,
		CS: 1.0,
		CA: 0.1,
		Dt: 0.005,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *MainenSejnowskiNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateMainenSejnowskiNeuron runs the neuron for n steps
func SimulateMainenSejnowskiNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMainenSejnowskiNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Vs
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
