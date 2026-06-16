// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for morris_lecar

package services

import (
	"math"
)

// MorrisLecarNeuronState holds the neuron state
type MorrisLecarNeuronState struct {
	V          float64
	W          float64
	CM         float64
	GCa        float64
	GK         float64
	GL         float64
	ECa        float64
	EK         float64
	EL         float64
	V1         float64
	V2         float64
	V3         float64
	V4         float64
	Phi        float64
	Dt         float64
	VThreshold float64
}

// NewMorrisLecarNeuron creates a new MorrisLecarNeuron neuron with default parameters
func NewMorrisLecarNeuron() *MorrisLecarNeuronState {
	return &MorrisLecarNeuronState{
		V:          -60.0,
		W:          0.0,
		CM:         20.0,
		GCa:        4.0,
		GK:         8.0,
		GL:         2.0,
		ECa:        120.0,
		EK:         -84.0,
		EL:         -60.0,
		V1:         -1.2,
		V2:         18.0,
		V3:         12.0,
		V4:         17.4,
		Phi:        1.0 / 15.0,
		Dt:         0.1,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *MorrisLecarNeuronState) Step(current float64) int {
	if !validateMorrisLecarState(s) || !finiteMorrisLecar(current) {
		return -1
	}
	vPrev := s.V
	k1V, k1W, ok := s.rhs(s.V, s.W, current)
	if !ok {
		return -1
	}
	k2V, k2W, ok := s.rhs(s.V+0.5*s.Dt*k1V, s.W+0.5*s.Dt*k1W, current)
	if !ok {
		return -1
	}
	k3V, k3W, ok := s.rhs(s.V+0.5*s.Dt*k2V, s.W+0.5*s.Dt*k2W, current)
	if !ok {
		return -1
	}
	k4V, k4W, ok := s.rhs(s.V+s.Dt*k3V, s.W+s.Dt*k3W, current)
	if !ok {
		return -1
	}
	next := *s
	next.V += s.Dt * (k1V + 2.0*k2V + 2.0*k3V + k4V) / 6.0
	next.W += s.Dt * (k1W + 2.0*k2W + 2.0*k3W + k4W) / 6.0
	if !validateMorrisLecarState(&next) {
		return -1
	}
	*s = next
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateMorrisLecarNeuron runs the neuron for n steps
func SimulateMorrisLecarNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMorrisLecarNeuron()
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

func (s *MorrisLecarNeuronState) mInf(v float64) float64 {
	return 0.5 * (1.0 + math.Tanh((v-s.V1)/s.V2))
}

func (s *MorrisLecarNeuronState) wInf(v float64) float64 {
	return 0.5 * (1.0 + math.Tanh((v-s.V3)/s.V4))
}

func (s *MorrisLecarNeuronState) lambda(v float64) float64 {
	return s.Phi * math.Cosh((v-s.V3)/(2.0*s.V4))
}

func (s *MorrisLecarNeuronState) rhs(v float64, w float64, current float64) (float64, float64, bool) {
	if !finiteMorrisLecar(v) || !finiteMorrisLecar(w) || !finiteMorrisLecar(current) || w < 0.0 || w > 1.0 {
		return 0.0, 0.0, false
	}
	mInf := s.mInf(v)
	wInf := s.wInf(v)
	lam := s.lambda(v)
	if !finiteMorrisLecar(mInf) || !finiteMorrisLecar(wInf) || !finiteMorrisLecar(lam) {
		return 0.0, 0.0, false
	}
	iCa := s.GCa * mInf * (v - s.ECa)
	iK := s.GK * w * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dv := (-iCa - iK - iL + current) / s.CM
	dw := lam * (wInf - w)
	if !finiteMorrisLecar(dv) || !finiteMorrisLecar(dw) {
		return 0.0, 0.0, false
	}
	return dv, dw, true
}

func finiteMorrisLecar(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validateMorrisLecarState(s *MorrisLecarNeuronState) bool {
	return finiteMorrisLecar(s.V) &&
		finiteMorrisLecar(s.W) &&
		finiteMorrisLecar(s.CM) &&
		finiteMorrisLecar(s.GCa) &&
		finiteMorrisLecar(s.GK) &&
		finiteMorrisLecar(s.GL) &&
		finiteMorrisLecar(s.ECa) &&
		finiteMorrisLecar(s.EK) &&
		finiteMorrisLecar(s.EL) &&
		finiteMorrisLecar(s.V1) &&
		finiteMorrisLecar(s.V2) &&
		finiteMorrisLecar(s.V3) &&
		finiteMorrisLecar(s.V4) &&
		finiteMorrisLecar(s.Phi) &&
		finiteMorrisLecar(s.Dt) &&
		finiteMorrisLecar(s.VThreshold) &&
		s.CM > 0.0 &&
		s.GCa > 0.0 &&
		s.GK > 0.0 &&
		s.GL > 0.0 &&
		s.V2 > 0.0 &&
		s.V4 > 0.0 &&
		s.Phi > 0.0 &&
		s.Dt > 0.0 &&
		s.W >= 0.0 &&
		s.W <= 1.0
}
