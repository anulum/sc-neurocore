// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go RK4 kernel for the VIP irregular-spiking neuron

package services

import (
	"math"
)

// VIPNeuronState holds the five-state VIP interneuron (Na, Kd, A-type K, leak).
type VIPNeuronState struct {
	V          float64
	H          float64
	N          float64
	A          float64
	B          float64
	GNa        float64
	GK         float64
	GA         float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Dt         float64
	VThreshold float64
}

// NewVIPNeuron creates a VIP neuron with the published defaults.
func NewVIPNeuron() *VIPNeuronState {
	return &VIPNeuronState{
		V: -65.0, H: 0.8, N: 0.1, A: 0.0, B: 0.9,
		GNa: 35.0, GK: 6.0, GA: 8.0, GL: 0.01,
		ENa: 55.0, EK: -90.0, EL: -65.0, CM: 0.5,
		Dt: 0.025, VThreshold: -20.0,
	}
}

// derivatives returns [dV, dh, dn, da, db] at one consistent state.
func (s *VIPNeuronState) derivatives(v, h, n, a, b, current float64) [5]float64 {
	mInf := 1.0 / (1.0 + math.Exp(-(v+30.0)/9.5))
	hInf := 1.0 / (1.0 + math.Exp((v+53.0)/7.0))
	tauH := 0.37 + 2.78/(1.0+math.Exp((v+40.5)/6.0))
	nInf := 1.0 / (1.0 + math.Exp(-(v+30.0)/10.0))
	tauN := 0.37 + 1.85/(1.0+math.Exp((v+27.0)/15.0))
	aInf := 1.0 / (1.0 + math.Exp(-(v+50.0)/20.0))
	bInf := 1.0 / (1.0 + math.Exp((v+78.0)/6.0))
	dh := (hInf - h) / tauH
	dn := (nInf - n) / tauN
	da := (aInf - a) / 5.0
	db := (bInf - b) / 50.0
	iNa := s.GNa * mInf * mInf * mInf * h * (v - s.ENa)
	iK := s.GK * n * n * n * n * (v - s.EK)
	iA := s.GA * a * a * a * b * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dv := (-iNa - iK - iA - iL + current) / s.CM
	return [5]float64{dv, dh, dn, da, db}
}

// rk4Substep returns one classical RK4 increment of (V, h, n, a, b).
func (s *VIPNeuronState) rk4Substep(st [5]float64, current float64) [5]float64 {
	dt := s.Dt
	k1 := s.derivatives(st[0], st[1], st[2], st[3], st[4], current)
	k2 := s.derivatives(st[0]+0.5*dt*k1[0], st[1]+0.5*dt*k1[1], st[2]+0.5*dt*k1[2], st[3]+0.5*dt*k1[3], st[4]+0.5*dt*k1[4], current)
	k3 := s.derivatives(st[0]+0.5*dt*k2[0], st[1]+0.5*dt*k2[1], st[2]+0.5*dt*k2[2], st[3]+0.5*dt*k2[3], st[4]+0.5*dt*k2[4], current)
	k4 := s.derivatives(st[0]+dt*k3[0], st[1]+dt*k3[1], st[2]+dt*k3[2], st[3]+dt*k3[3], st[4]+dt*k3[4], current)
	var out [5]float64
	for i := 0; i < 5; i++ {
		out[i] = st[i] + dt*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])/6.0
	}
	return out
}

// Step advances the neuron by one 4*dt step and reports an upward threshold crossing.
func (s *VIPNeuronState) Step(iExt float64) int {
	vPrev := s.V
	st := [5]float64{s.V, s.H, s.N, s.A, s.B}
	for i := 0; i < 4; i++ {
		st = s.rk4Substep(st, iExt)
	}
	s.V, s.H, s.N, s.A, s.B = st[0], st[1], st[2], st[3], st[4]
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateVIPNeuron runs the neuron for nSteps and returns (trace, spikes).
func SimulateVIPNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewVIPNeuron()
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
