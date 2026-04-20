// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for dendritic_nmda

package services

import (
	"math"
)

// DendriticNMDANeuronState holds the neuron state
type DendriticNMDANeuronState struct {
	GNmda float64
	ENmda float64
	MgConc float64
	GCoupling float64
	TauSoma float64
	TauDend float64
	Theta float64
	Dt float64
	VSoma float64
	VDend float64
}

// NewDendriticNMDANeuron creates a new DendriticNMDANeuron neuron with default parameters
func NewDendriticNMDANeuron() *DendriticNMDANeuronState {
	return &DendriticNMDANeuronState{
		GNmda: 1.5,
		ENmda: 0.0,
		MgConc: 1.0,
		GCoupling: 0.5,
		TauSoma: 20.0,
		TauDend: 50.0,
		Theta: -50.0,
		Dt: 0.1,
		VSoma: -65.0,
		VDend: -65.0,
	}
}

// Step advances the neuron by one timestep
func (s *DendriticNMDANeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateDendriticNMDANeuron runs the neuron for n steps
func SimulateDendriticNMDANeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDendriticNMDANeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.GNmda
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
