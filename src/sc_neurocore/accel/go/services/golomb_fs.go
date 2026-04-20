// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for golomb_fs

package services

import (
	"math"
)

// GolombFSNeuronState holds the neuron state
type GolombFSNeuronState struct {
	V float64
	H float64
	N float64
	P float64
	GNa float64
	GKd float64
	GKv3 float64
	GL float64
	ENa float64
	EK float64
	EL float64
	CM float64
	Dt float64
	VThreshold float64
}

// NewGolombFSNeuron creates a new GolombFSNeuron neuron with default parameters
func NewGolombFSNeuron() *GolombFSNeuronState {
	return &GolombFSNeuronState{
		V: -65.0,
		H: 0.9,
		N: 0.1,
		P: 0.0,
		GNa: 112.5,
		GKd: 225.0,
		GKv3: 150.0,
		GL: 0.25,
		ENa: 50.0,
		EK: -90.0,
		EL: -70.0,
		CM: 1.0,
		Dt: 0.01,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *GolombFSNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateGolombFSNeuron runs the neuron for n steps
func SimulateGolombFSNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGolombFSNeuron()
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
