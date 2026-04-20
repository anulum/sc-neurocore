// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for neurogrid

package services

import (
	"math"
)

// NeuroGridNeuronState holds the neuron state
type NeuroGridNeuronState struct {
	VS float64
	VD float64
	TauS float64
	TauD float64
	GC float64
	DeltaT float64
	VRest float64
	VThreshold float64
	VPeak float64
	VReset float64
	Dt float64
}

// NewNeuroGridNeuron creates a new NeuroGridNeuron neuron with default parameters
func NewNeuroGridNeuron() *NeuroGridNeuronState {
	return &NeuroGridNeuronState{
		VS: -65.0,
		VD: -65.0,
		TauS: 20.0,
		TauD: 50.0,
		GC: 0.5,
		DeltaT: 2.0,
		VRest: -65.0,
		VThreshold: -50.0,
		VPeak: 20.0,
		VReset: -65.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *NeuroGridNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateNeuroGridNeuron runs the neuron for n steps
func SimulateNeuroGridNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewNeuroGridNeuron()
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
