// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for gutkin_ermentrout

package services

import (
	"math"
)

// GutkinErmentroutNeuronState holds the neuron state
type GutkinErmentroutNeuronState struct {
	V          float64
	N          float64
	GNa        float64
	GK         float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	Dt         float64
	VThreshold float64
}

// NewGutkinErmentroutNeuron creates a new GutkinErmentroutNeuron neuron with default parameters
func NewGutkinErmentroutNeuron() *GutkinErmentroutNeuronState {
	return &GutkinErmentroutNeuronState{
		V:          -65.0,
		N:          0.1,
		GNa:        20.0,
		GK:         10.0,
		GL:         8.0,
		ENa:        60.0,
		EK:         -90.0,
		EL:         -80.0,
		Dt:         0.05,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep
func (s *GutkinErmentroutNeuronState) Step(iExt float64) int {
	if !validateGutkinErmentroutState(s) || !finiteGutkinErmentrout(iExt) {
		return -1
	}
	vPrev := s.V
	k1V, k1N, ok := s.rhs(s.V, s.N, iExt)
	if !ok {
		return -1
	}
	k2V, k2N, ok := s.rhs(s.V+0.5*s.Dt*k1V, s.N+0.5*s.Dt*k1N, iExt)
	if !ok {
		return -1
	}
	k3V, k3N, ok := s.rhs(s.V+0.5*s.Dt*k2V, s.N+0.5*s.Dt*k2N, iExt)
	if !ok {
		return -1
	}
	k4V, k4N, ok := s.rhs(s.V+s.Dt*k3V, s.N+s.Dt*k3N, iExt)
	if !ok {
		return -1
	}
	next := *s
	next.V += s.Dt * (k1V + 2.0*k2V + 2.0*k3V + k4V) / 6.0
	next.N += s.Dt * (k1N + 2.0*k2N + 2.0*k3N + k4N) / 6.0
	if !validateGutkinErmentroutState(&next) {
		return -1
	}
	*s = next
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateGutkinErmentroutNeuron runs the neuron for n steps
func SimulateGutkinErmentroutNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGutkinErmentroutNeuron()
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

func (s *GutkinErmentroutNeuronState) mInf(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-(v+20.0)/15.0))
}

func (s *GutkinErmentroutNeuronState) nInf(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-(v+25.0)/5.0))
}

func (s *GutkinErmentroutNeuronState) rhs(v float64, nGate float64, iExt float64) (float64, float64, bool) {
	if !finiteGutkinErmentrout(v) || !finiteGutkinErmentrout(nGate) || !finiteGutkinErmentrout(iExt) || nGate < 0.0 || nGate > 1.0 {
		return 0.0, 0.0, false
	}
	mInf := s.mInf(v)
	nInf := s.nInf(v)
	if !finiteGutkinErmentrout(mInf) || !finiteGutkinErmentrout(nInf) {
		return 0.0, 0.0, false
	}
	iNa := s.GNa * mInf * (v - s.ENa)
	iK := s.GK * nGate * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dv := -iNa - iK - iL + iExt
	dn := nInf - nGate
	if !finiteGutkinErmentrout(dv) || !finiteGutkinErmentrout(dn) {
		return 0.0, 0.0, false
	}
	return dv, dn, true
}

func finiteGutkinErmentrout(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validateGutkinErmentroutState(s *GutkinErmentroutNeuronState) bool {
	return finiteGutkinErmentrout(s.V) &&
		finiteGutkinErmentrout(s.N) &&
		s.N >= 0.0 &&
		s.N <= 1.0 &&
		finiteGutkinErmentrout(s.GNa) &&
		s.GNa >= 0.0 &&
		finiteGutkinErmentrout(s.GK) &&
		s.GK >= 0.0 &&
		finiteGutkinErmentrout(s.GL) &&
		s.GL >= 0.0 &&
		finiteGutkinErmentrout(s.ENa) &&
		finiteGutkinErmentrout(s.EK) &&
		finiteGutkinErmentrout(s.EL) &&
		finiteGutkinErmentrout(s.Dt) &&
		s.Dt > 0.0 &&
		finiteGutkinErmentrout(s.VThreshold)
}
