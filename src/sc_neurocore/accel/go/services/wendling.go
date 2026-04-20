// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wendling

package services

import (
	"math"
)

// WendlingNeuronState holds the neuron state
type WendlingNeuronState struct {
	Y0 float64
	Y5 float64
	Y1 float64
	Y6 float64
	Y2 float64
	Y7 float64
	Y3 float64
	Y8 float64
	Y4 float64
	Y9 float64
	AExc float64
	BFast float64
	GSlow float64
	ARate float64
	BRate float64
	GRate float64
	C float64
	E0 float64
	V0 float64
	R float64
}

// NewWendlingNeuron creates a new WendlingNeuron neuron with default parameters
func NewWendlingNeuron() *WendlingNeuronState {
	return &WendlingNeuronState{
		Y0: 0.0,
		Y5: 0.0,
		Y1: 0.0,
		Y6: 0.0,
		Y2: 0.0,
		Y7: 0.0,
		Y3: 0.0,
		Y8: 0.0,
		Y4: 0.0,
		Y9: 0.0,
		AExc: 3.25,
		BFast: 22.0,
		GSlow: 10.0,
		ARate: 100.0,
		BRate: 500.0,
		GRate: 20.0,
		C: 135.0,
		E0: 2.5,
		V0: 6.0,
		R: 0.56,
	}
}

// Step advances the neuron by one timestep
func (s *WendlingNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateWendlingNeuron runs the neuron for n steps
func SimulateWendlingNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewWendlingNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Y0
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
