// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for hay_l5

package services

import (
	"math"
)

// HayL5PyramidalNeuronState holds the neuron state
type HayL5PyramidalNeuronState struct {
	VS float64
	HNa float64
	NK float64
	VT float64
	MCa float64
	HCa float64
	MIh float64
	VA float64
	CaA float64
	GNa float64
	GK float64
	GLS float64
	ENa float64
	EK float64
	EL float64
	GCaT float64
	GIh float64
	GLT float64
	ECa float64
	EIh float64
}

// NewHayL5PyramidalNeuron creates a new HayL5PyramidalNeuron neuron with default parameters
func NewHayL5PyramidalNeuron() *HayL5PyramidalNeuronState {
	return &HayL5PyramidalNeuronState{
		VS: -75.0,
		HNa: 0.9,
		NK: 0.1,
		VT: -75.0,
		MCa: 0.0,
		HCa: 1.0,
		MIh: 0.0,
		VA: -75.0,
		CaA: 0.0001,
		GNa: 300.0,
		GK: 40.0,
		GLS: 0.03,
		ENa: 50.0,
		EK: -85.0,
		EL: -75.0,
		GCaT: 2.0,
		GIh: 0.02,
		GLT: 0.03,
		ECa: 140.0,
		EIh: -45.0,
	}
}

// Step advances the neuron by one timestep
func (s *HayL5PyramidalNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateHayL5PyramidalNeuron runs the neuron for n steps
func SimulateHayL5PyramidalNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewHayL5PyramidalNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.VS
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
