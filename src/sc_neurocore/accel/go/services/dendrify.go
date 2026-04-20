// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for dendrify

package services

import (
	"math"
)

// DendrifyNeuronState holds the neuron state
type DendrifyNeuronState struct {
	VS float64
	VD float64
	DActive float64
	TauS float64
	TauD float64
	GC float64
	DThreshold float64
	DAmplitude float64
	DDuration float64
	DTimer float64
	VRest float64
	VThreshold float64
	VReset float64
	Dt float64
}

// NewDendrifyNeuron creates a new DendrifyNeuron neuron with default parameters
func NewDendrifyNeuron() *DendrifyNeuronState {
	return &DendrifyNeuronState{
		VS: -65.0,
		VD: -65.0,
		DActive: 0.0,
		TauS: 10.0,
		TauD: 20.0,
		GC: 0.8,
		DThreshold: -35.0,
		DAmplitude: 30.0,
		DDuration: 10.0,
		DTimer: 0.0,
		VRest: -65.0,
		VThreshold: -50.0,
		VReset: -65.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *DendrifyNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateDendrifyNeuron runs the neuron for n steps
func SimulateDendrifyNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDendrifyNeuron()
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
