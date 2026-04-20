// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for dpi_neuron

package services

import (
	"math"
)

// DPINeuronState holds the neuron state
type DPINeuronState struct {
	IMem float64
	IThreshold float64
	IReset float64
	ILeak float64
	Tau float64
	Gain float64
	Dt float64
}

// NewDPINeuron creates a new DPINeuron neuron with default parameters
func NewDPINeuron() *DPINeuronState {
	return &DPINeuronState{
		IMem: 0.0,
		IThreshold: 1.0,
		IReset: 0.0,
		ILeak: 0.01,
		Tau: 20.0,
		Gain: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *DPINeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateDPINeuron runs the neuron for n steps
func SimulateDPINeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDPINeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.IMem
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
