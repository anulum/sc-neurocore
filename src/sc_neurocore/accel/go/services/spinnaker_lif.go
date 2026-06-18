// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for spinnaker_lif

package services

import (
	"math"
)

// SpiNNakerLIFNeuronState holds the exact-flow SpiNNaker LIF neuron state.
type SpiNNakerLIFNeuronState struct {
	V           float64
	VRest       float64
	VReset      float64
	VThreshold  float64
	TauM        float64
	IOffset     float64
	TauRefrac   float64
	RefracCount float64
	Dt          float64
}

// NewSpiNNakerLIFNeuron creates a SpiNNaker LIF neuron with documented defaults.
func NewSpiNNakerLIFNeuron() *SpiNNakerLIFNeuronState {
	return &SpiNNakerLIFNeuronState{
		V:           -70.0,
		VRest:       -70.0,
		VReset:      -70.0,
		VThreshold:  -50.0,
		TauM:        20.0,
		IOffset:     0.0,
		TauRefrac:   2.0,
		RefracCount: 0.0,
		Dt:          1.0,
	}
}

// Step advances the neuron by one exact-flow timestep.
func (s *SpiNNakerLIFNeuronState) Step(iExt float64) int {
	if !s.Valid() || !isFiniteSpiNNakerLIF(iExt) {
		return -1
	}
	if s.RefracCount > 0.0 {
		s.RefracCount = math.Max(0.0, s.RefracCount-s.Dt)
		return 0
	}
	steady := s.VRest + iExt + s.IOffset
	nextV := steady + (s.V-steady)*math.Exp(-s.Dt/s.TauM)
	if !isFiniteSpiNNakerLIF(nextV) {
		return -1
	}
	if nextV >= s.VThreshold {
		s.V = s.VReset
		s.RefracCount = s.TauRefrac
		return 1
	}
	s.V = nextV
	return 0
}

// Reset restores the documented rest state and clears the refractory timer.
func (s *SpiNNakerLIFNeuronState) Reset() {
	s.V = s.VRest
	s.RefracCount = 0.0
}

// Valid reports whether the state satisfies the exact-flow scalar contract.
func (s *SpiNNakerLIFNeuronState) Valid() bool {
	return isFiniteSpiNNakerLIF(s.V) &&
		isFiniteSpiNNakerLIF(s.VRest) &&
		isFiniteSpiNNakerLIF(s.VReset) &&
		isFiniteSpiNNakerLIF(s.VThreshold) &&
		isFiniteSpiNNakerLIF(s.TauM) &&
		s.TauM > 0.0 &&
		isFiniteSpiNNakerLIF(s.IOffset) &&
		isFiniteSpiNNakerLIF(s.TauRefrac) &&
		s.TauRefrac >= 0.0 &&
		isFiniteSpiNNakerLIF(s.RefracCount) &&
		s.RefracCount >= 0.0 &&
		isFiniteSpiNNakerLIF(s.Dt) &&
		s.Dt > 0.0 &&
		s.VThreshold > s.VReset
}

func isFiniteSpiNNakerLIF(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// SimulateSpiNNakerLIFNeuron runs the neuron for n steps.
func SimulateSpiNNakerLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSpiNNakerLIFNeuron()
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
