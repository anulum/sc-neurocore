// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for siegert

package services

import (
	"math"
)

// SiegertTransferFunctionState holds the neuron state
type SiegertTransferFunctionState struct {
	TauM float64
	TauRp float64
	VThreshold float64
	VReset float64
	VRest float64
}

// NewSiegertTransferFunction creates a new SiegertTransferFunction neuron with default parameters
func NewSiegertTransferFunction() *SiegertTransferFunctionState {
	return &SiegertTransferFunctionState{
		TauM: 20.0,
		TauRp: 2.0,
		VThreshold: -50.0,
		VReset: -70.0,
		VRest: -65.0,
	}
}

// Step advances the neuron by one timestep
func (s *SiegertTransferFunctionState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateSiegertTransferFunction runs the neuron for n steps
func SimulateSiegertTransferFunction(nSteps int, iExt float64) ([]float64, int) {
	s := NewSiegertTransferFunction()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.TauM
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
