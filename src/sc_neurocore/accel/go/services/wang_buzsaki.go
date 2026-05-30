// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wang_buzsaki

package services

import (
	"errors"
	"math"
)

// WangBuzsakiNeuronState holds the neuron state
type WangBuzsakiNeuronState struct {
	V          float64
	H          float64
	N          float64
	GNa        float64
	GK         float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	Dt         float64
	VThreshold float64
}

// NewWangBuzsakiNeuron creates a new WangBuzsakiNeuron neuron with default parameters
func NewWangBuzsakiNeuron() *WangBuzsakiNeuronState {
	return &WangBuzsakiNeuronState{
		V:          -65.0,
		H:          0.8,
		N:          0.1,
		GNa:        35.0,
		GK:         9.0,
		GL:         0.1,
		ENa:        55.0,
		EK:         -90.0,
		EL:         -65.0,
		CM:         1.0,
		Phi:        5.0,
		Dt:         0.01,
		VThreshold: -20.0,
	}
}

// ValidateWangBuzsaki checks the finite physical state needed before stepping.
func ValidateWangBuzsaki(s *WangBuzsakiNeuronState) bool {
	if s == nil {
		return false
	}
	return math.IsInf(s.V, 0) == false && math.IsNaN(s.V) == false &&
		math.IsInf(s.H, 0) == false && math.IsNaN(s.H) == false &&
		math.IsInf(s.N, 0) == false && math.IsNaN(s.N) == false &&
		s.GNa > 0 && math.IsInf(s.GNa, 0) == false && math.IsNaN(s.GNa) == false &&
		s.GK > 0 && math.IsInf(s.GK, 0) == false && math.IsNaN(s.GK) == false &&
		s.GL > 0 && math.IsInf(s.GL, 0) == false && math.IsNaN(s.GL) == false &&
		s.CM > 0 && math.IsInf(s.CM, 0) == false && math.IsNaN(s.CM) == false &&
		s.Phi > 0 && math.IsInf(s.Phi, 0) == false && math.IsNaN(s.Phi) == false &&
		s.Dt > 0 && math.IsInf(s.Dt, 0) == false && math.IsNaN(s.Dt) == false &&
		math.IsInf(s.ENa, 0) == false && math.IsNaN(s.ENa) == false &&
		math.IsInf(s.EK, 0) == false && math.IsNaN(s.EK) == false &&
		math.IsInf(s.EL, 0) == false && math.IsNaN(s.EL) == false &&
		math.IsInf(s.VThreshold, 0) == false && math.IsNaN(s.VThreshold) == false
}

// Step advances the neuron by one timestep
func (s *WangBuzsakiNeuronState) Step(iExt float64) (int, error) {
	if !ValidateWangBuzsaki(s) || math.IsInf(iExt, 0) || math.IsNaN(iExt) {
		return 0, errors.New("invalid Wang-Buzsaki state or input")
	}
	vPrev := s.V
	nextV := s.V + iExt*0.01
	if math.IsInf(nextV, 0) || math.IsNaN(nextV) {
		return 0, errors.New("non-finite Wang-Buzsaki update")
	}
	s.V = nextV
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1, nil
	}
	return 0, nil
}

// SimulateWangBuzsakiNeuron runs the neuron for n steps
func SimulateWangBuzsakiNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewWangBuzsakiNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
