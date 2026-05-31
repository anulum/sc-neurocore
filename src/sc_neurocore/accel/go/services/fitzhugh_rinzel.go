// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for fitzhugh_rinzel

package services

import (
	"math"
)

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// FitzHughRinzelNeuronState holds the neuron state
type FitzHughRinzelNeuronState struct {
	V          float64
	W          float64
	Y          float64
	A          float64
	B          float64
	C          float64
	D          float64
	Delta      float64
	Mu         float64
	Dt         float64
	VThreshold float64
}

// NewFitzHughRinzelNeuron creates a new FitzHughRinzelNeuron neuron with default parameters
func NewFitzHughRinzelNeuron() *FitzHughRinzelNeuronState {
	return &FitzHughRinzelNeuronState{
		V:          -1.0,
		W:          -0.5,
		Y:          0.0,
		A:          0.7,
		B:          0.8,
		C:          -0.775,
		D:          1.0,
		Delta:      0.08,
		Mu:         0.0001,
		Dt:         0.1,
		VThreshold: 1.0,
	}
}

func (s *FitzHughRinzelNeuronState) valid() bool {
	return isFinite(s.V) && isFinite(s.W) && isFinite(s.Y) &&
		isFinite(s.A) && isFinite(s.B) && isFinite(s.C) && isFinite(s.D) &&
		isFinite(s.Delta) && isFinite(s.Mu) && isFinite(s.Dt) && isFinite(s.VThreshold) &&
		s.Delta > 0 && s.Mu > 0 && s.Dt > 0
}

func (s *FitzHughRinzelNeuronState) derivatives(iExt float64) (float64, float64, float64, bool) {
	if !isFinite(iExt) {
		return 0, 0, 0, false
	}
	dv := s.V - s.V*s.V*s.V/3.0 - s.W + s.Y + iExt
	dw := s.Delta * (s.A + s.V - s.B*s.W)
	dy := s.Mu * (s.C - s.V - s.D*s.Y)
	return dv, dw, dy, isFinite(dv) && isFinite(dw) && isFinite(dy)
}

// Step advances the neuron by one timestep.
func (s *FitzHughRinzelNeuronState) Step(iExt float64) int {
	if !s.valid() {
		return 0
	}
	vPrev := s.V
	dv, dw, dy, ok := s.derivatives(iExt)
	if !ok {
		return 0
	}
	nextV := s.V + dv*s.Dt
	nextW := s.W + dw*s.Dt
	nextY := s.Y + dy*s.Dt
	if !(isFinite(nextV) && isFinite(nextW) && isFinite(nextY)) {
		return 0
	}
	s.V = nextV
	s.W = nextW
	s.Y = nextY
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateFitzHughRinzelNeuron runs the neuron for n steps
func SimulateFitzHughRinzelNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewFitzHughRinzelNeuron()
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
