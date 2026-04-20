// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for hodgkin_huxley

package services

import (
	"math"
)

// HodgkinHuxleyNeuronState holds the neuron state
type HodgkinHuxleyNeuronState struct {
	V float64
	M float64
	H float64
	N float64
	CM float64
	GNa float64
	GK float64
	GL float64
	ENa float64
	EK float64
	EL float64
	Dt float64
	VThreshold float64
}

// NewHodgkinHuxleyNeuron creates a new HodgkinHuxleyNeuron neuron with default parameters
func NewHodgkinHuxleyNeuron() *HodgkinHuxleyNeuronState {
	return &HodgkinHuxleyNeuronState{
		V: -65.0,
		M: 0.05,
		H: 0.6,
		N: 0.32,
		CM: 1.0,
		GNa: 120.0,
		GK: 36.0,
		GL: 0.3,
		ENa: 50.0,
		EK: -77.0,
		EL: -54.4,
		Dt: 0.01,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *HodgkinHuxleyNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateHodgkinHuxleyNeuron runs the neuron for n steps
func SimulateHodgkinHuxleyNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewHodgkinHuxleyNeuron()
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
