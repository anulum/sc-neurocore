// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for plant_r15

package services

import (
	"math"
)

// PlantR15NeuronState holds the neuron state
type PlantR15NeuronState struct {
	V float64
	M float64
	H float64
	N float64
	Ca float64
	GNa float64
	GK float64
	GCa float64
	GL float64
	GKca float64
	ENa float64
	EK float64
	ECa float64
	EL float64
	CM float64
	KCa float64
	TauCa float64
	Dt float64
	VThreshold float64
}

// NewPlantR15Neuron creates a new PlantR15Neuron neuron with default parameters
func NewPlantR15Neuron() *PlantR15NeuronState {
	return &PlantR15NeuronState{
		V: -50.0,
		M: 0.05,
		H: 0.6,
		N: 0.3,
		Ca: 0.1,
		GNa: 4.0,
		GK: 0.3,
		GCa: 0.004,
		GL: 0.003,
		GKca: 0.03,
		ENa: 30.0,
		EK: -75.0,
		ECa: 140.0,
		EL: -40.0,
		CM: 1.0,
		KCa: 0.0085,
		TauCa: 500.0,
		Dt: 0.05,
		VThreshold: -10.0,
	}
}

// Step advances the neuron by one timestep
func (s *PlantR15NeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -50.0
		return 1
	}
	return 0
}

// SimulatePlantR15Neuron runs the neuron for n steps
func SimulatePlantR15Neuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPlantR15Neuron()
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
