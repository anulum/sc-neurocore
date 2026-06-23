// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go RK4 kernel for the Golomb et al. 2007 fast-spiking neuron

package services

import (
	"math"
)

// GolombFSNeuronState holds the four-state Golomb et al. 2007 fast-spiking neuron.
type GolombFSNeuronState struct {
	V          float64
	H          float64
	N          float64
	P          float64
	GNa        float64
	GKd        float64
	GKv3       float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Dt         float64
	VThreshold float64
}

// NewGolombFSNeuron creates a Golomb-FS neuron with the published defaults.
func NewGolombFSNeuron() *GolombFSNeuronState {
	return &GolombFSNeuronState{
		V: -65.0, H: 0.9, N: 0.1, P: 0.0,
		GNa: 112.5, GKd: 225.0, GKv3: 150.0, GL: 0.25,
		ENa: 50.0, EK: -90.0, EL: -70.0, CM: 1.0,
		Dt: 0.01, VThreshold: -20.0,
	}
}

// derivatives returns [dV, dh, dn, dp] at one consistent state.
func (s *GolombFSNeuronState) derivatives(v, h, n, p, current float64) [4]float64 {
	mInf := 1.0 / (1.0 + math.Exp(-(v+24.0)/11.5))
	hInf := 1.0 / (1.0 + math.Exp((v+58.3)/6.7))
	tauH := 0.5 + 14.0/(1.0+math.Exp((v+60.0)/12.0))
	nInf := 1.0 / (1.0 + math.Exp(-(v+12.4)/6.8))
	tauN := 0.087 + 11.4/(1.0+math.Exp((v+14.6)/8.6))
	pInf := 1.0 / (1.0 + math.Exp(-(v+3.0)/8.0))
	tauP := 0.1 + 4.0/(1.0+math.Exp((v+25.0)/10.0))
	dh := (hInf - h) / tauH
	dn := (nInf - n) / tauN
	dp := (pInf - p) / tauP
	iNa := s.GNa * mInf * mInf * mInf * h * (v - s.ENa)
	iKd := s.GKd * n * n * n * n * (v - s.EK)
	iKv3 := s.GKv3 * p * p * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dv := (-iNa - iKd - iKv3 - iL + current) / s.CM
	return [4]float64{dv, dh, dn, dp}
}

// rk4Substep returns one classical RK4 increment of (V, h, n, p).
func (s *GolombFSNeuronState) rk4Substep(st [4]float64, current float64) [4]float64 {
	dt := s.Dt
	k1 := s.derivatives(st[0], st[1], st[2], st[3], current)
	k2 := s.derivatives(st[0]+0.5*dt*k1[0], st[1]+0.5*dt*k1[1], st[2]+0.5*dt*k1[2], st[3]+0.5*dt*k1[3], current)
	k3 := s.derivatives(st[0]+0.5*dt*k2[0], st[1]+0.5*dt*k2[1], st[2]+0.5*dt*k2[2], st[3]+0.5*dt*k2[3], current)
	k4 := s.derivatives(st[0]+dt*k3[0], st[1]+dt*k3[1], st[2]+dt*k3[2], st[3]+dt*k3[3], current)
	var out [4]float64
	for i := 0; i < 4; i++ {
		out[i] = st[i] + dt*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])/6.0
	}
	return out
}

// Step advances the neuron by one 10*dt step and reports an upward threshold crossing.
func (s *GolombFSNeuronState) Step(iExt float64) int {
	vPrev := s.V
	st := [4]float64{s.V, s.H, s.N, s.P}
	for i := 0; i < 10; i++ {
		st = s.rk4Substep(st, iExt)
	}
	s.V, s.H, s.N, s.P = st[0], st[1], st[2], st[3]
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateGolombFSNeuron runs the neuron for nSteps and returns (trace, spikes).
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
