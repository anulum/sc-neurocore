// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go RK4 kernel for the Pospischil et al. 2008 neuron

package services

import (
	"math"
)

// PospischilNeuronState holds the five-state Pospischil et al. 2008 neuron.
type PospischilNeuronState struct {
	V          float64
	M          float64
	H          float64
	N          float64
	P          float64
	GNa        float64
	GKd        float64
	GM         float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Vt         float64
	Dt         float64
	VThreshold float64
}

// NewPospischilNeuron creates a Pospischil neuron with the regular-spiking defaults.
func NewPospischilNeuron() *PospischilNeuronState {
	return &PospischilNeuronState{
		V: -70.0, M: 0.05, H: 0.6, N: 0.3, P: 0.0,
		GNa: 50.0, GKd: 5.0, GM: 0.07, GL: 0.1,
		ENa: 50.0, EK: -90.0, EL: -70.0, CM: 1.0,
		Vt: -56.2, Dt: 0.025, VThreshold: -20.0,
	}
}

// alphaSingular returns num/(exp(num/slope)-1) with the closed-form L'Hôpital
// limit within 1e-6 of the removable singularity, matching the other kernels.
func alphaSingular(num, slope, limit float64) float64 {
	if math.Abs(num) < 1e-6 {
		return limit
	}
	return num / (math.Exp(num/slope) - 1.0)
}

// derivatives returns [dV, dm, dh, dn, dp] at one consistent state.
func (s *PospischilNeuronState) derivatives(v, m, h, n, p, current float64) [5]float64 {
	dvVt := v - s.Vt
	am := -0.32 * alphaSingular(dvVt-13.0, -4.0, -4.0)
	bm := 0.28 * alphaSingular(dvVt-40.0, 5.0, 5.0)
	ah := 0.128 * math.Exp(-(dvVt-17.0)/18.0)
	bh := 4.0 / (1.0 + math.Exp(-(dvVt-40.0)/5.0))
	an := -0.032 * alphaSingular(dvVt-15.0, -5.0, -5.0)
	bn := 0.5 * math.Exp(-(dvVt-10.0)/40.0)
	pInf := 1.0 / (1.0 + math.Exp(-(v+35.0)/10.0))
	tauP := 608.0 / (3.3*math.Exp((v+35.0)/20.0) + math.Exp(-(v+35.0)/20.0))
	dm := am*(1.0-m) - bm*m
	dh := ah*(1.0-h) - bh*h
	dn := an*(1.0-n) - bn*n
	dp := (pInf - p) / tauP
	iNa := s.GNa * m * m * m * h * (v - s.ENa)
	iKd := s.GKd * n * n * n * n * (v - s.EK)
	iM := s.GM * p * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dv := (-iNa - iKd - iM - iL + current) / s.CM
	return [5]float64{dv, dm, dh, dn, dp}
}

// rk4Substep returns one classical RK4 increment of (V, m, h, n, p).
func (s *PospischilNeuronState) rk4Substep(st [5]float64, current float64) [5]float64 {
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
func (s *PospischilNeuronState) Step(iExt float64) int {
	vPrev := s.V
	st := [5]float64{s.V, s.M, s.H, s.N, s.P}
	for i := 0; i < 4; i++ {
		st = s.rk4Substep(st, iExt)
	}
	s.V, s.M, s.H, s.N, s.P = st[0], st[1], st[2], st[3], st[4]
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulatePospischilNeuron runs the neuron for nSteps and returns (trace, spikes).
func SimulatePospischilNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPospischilNeuron()
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
