// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for brainscales_adex

package services

import (
	"math"
)

// BrainScaleSAdExNeuronState holds the neuron state
type BrainScaleSAdExNeuronState struct {
	V float64
	W float64
	VRest float64
	VReset float64
	VThreshold float64
	DeltaT float64
	VRh float64
	Tau float64
	TauW float64
	A float64
	B float64
	HwSpeedup float64
	Dt float64
}

// NewBrainScaleSAdExNeuron creates a new BrainScaleSAdExNeuron neuron with default parameters
func NewBrainScaleSAdExNeuron() *BrainScaleSAdExNeuronState {
	return &BrainScaleSAdExNeuronState{
		V: -65.0,
		W: 0.0,
		VRest: -65.0,
		VReset: -68.0,
		VThreshold: -50.0,
		DeltaT: 2.0,
		VRh: -55.0,
		Tau: 20.0,
		TauW: 100.0,
		A: 0.5,
		B: 7.0,
		HwSpeedup: 1000.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *BrainScaleSAdExNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateBrainScaleSAdExNeuron runs the neuron for n steps
func SimulateBrainScaleSAdExNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewBrainScaleSAdExNeuron()
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
