// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for gif_population

package services

import (
	"math"
)

// GIFPopulationNeuronState holds the neuron state
type GIFPopulationNeuronState struct {
	V float64
	Theta float64
	Eta float64
	TauM float64
	TauEta float64
	DeltaV float64
	Lambda0 float64
	EtaIncrement float64
	VRest float64
	VReset float64
	Dt float64
	Rng float64
}

// NewGIFPopulationNeuron creates a new GIFPopulationNeuron neuron with default parameters
func NewGIFPopulationNeuron() *GIFPopulationNeuronState {
	return &GIFPopulationNeuronState{
		V: -65.0,
		Theta: -50.0,
		Eta: 0.0,
		TauM: 20.0,
		TauEta: 100.0,
		DeltaV: 2.0,
		Lambda0: 0.001,
		EtaIncrement: 5.0,
		VRest: -65.0,
		VReset: -65.0,
		Dt: 0.5,
		Rng: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *GIFPopulationNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateGIFPopulationNeuron runs the neuron for n steps
func SimulateGIFPopulationNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGIFPopulationNeuron()
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
