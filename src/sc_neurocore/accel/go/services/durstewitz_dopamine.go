// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for durstewitz_dopamine

package services

import (
	"math"
)

// DurstewitzDopamineNeuronState holds the neuron state
type DurstewitzDopamineNeuronState struct {
	V float64
	HNa float64
	NK float64
	GNa float64
	GK float64
	GNmda float64
	GL float64
	ENa float64
	EK float64
	ENmda float64
	EL float64
	Mg float64
	D1Level float64
	GNmdaScale float64
	GKScale float64
	VShiftNa float64
	Dt float64
	VThreshold float64
}

// NewDurstewitzDopamineNeuron creates a new DurstewitzDopamineNeuron neuron with default parameters
func NewDurstewitzDopamineNeuron() *DurstewitzDopamineNeuronState {
	return &DurstewitzDopamineNeuronState{
		V: -65.0,
		HNa: 0.7,
		NK: 0.2,
		GNa: 45.0,
		GK: 18.0,
		GNmda: 0.5,
		GL: 0.02,
		ENa: 55.0,
		EK: -80.0,
		ENmda: 0.0,
		EL: -65.0,
		Mg: 1.0,
		D1Level: 0.0,
		GNmdaScale: 2.5,
		GKScale: 1.5,
		VShiftNa: -5.0,
		Dt: 0.05,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *DurstewitzDopamineNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateDurstewitzDopamineNeuron runs the neuron for n steps
func SimulateDurstewitzDopamineNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDurstewitzDopamineNeuron()
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
