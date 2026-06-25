// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go RK4 kernel for the SST+ low-threshold spiking neuron

package services

import (
	"math"
)

// SSTNeuronState holds the seven-state SST+ LTS interneuron (Na, Kd, M, T-Ca, Ih, leak).
type SSTNeuronState struct {
	V          float64
	M          float64
	H          float64
	N          float64
	P          float64
	S          float64
	R          float64
	GNa        float64
	GK         float64
	GM         float64
	GT         float64
	GH         float64
	GL         float64
	ENa        float64
	EK         float64
	ECa        float64
	EH         float64
	EL         float64
	CM         float64
	Dt         float64
	VThreshold float64
}

// NewSSTNeuron creates an SST neuron with the Pospischil 2008 LTS defaults.
func NewSSTNeuron() *SSTNeuronState {
	return &SSTNeuronState{
		V: -65.0, M: 0.02, H: 0.8, N: 0.2, P: 0.0, S: 0.9, R: 0.1,
		GNa: 50.0, GK: 5.0, GM: 0.12, GT: 0.01, GH: 0.02, GL: 0.05,
		ENa: 50.0, EK: -90.0, ECa: 120.0, EH: -40.0, EL: -65.0,
		CM: 1.0, Dt: 0.025, VThreshold: -20.0,
	}
}

// asing returns the L'Hôpital limit of the Traub-Miles x/(exp(x/slope)-1) rate.
func asing(num, slope, limit float64) float64 {
	if math.Abs(num) < 1e-6 {
		return limit
	}
	return num / (math.Exp(num/slope) - 1.0)
}

// derivatives returns [dV, dm, dh, dn, dp, ds, dr] at one consistent state.
func (s *SSTNeuronState) derivatives(v, m, h, n, p, sg, r, current float64) [7]float64 {
	dvt := v - (-56.2)
	alphaM := -0.32 * asing(dvt-13.0, -4.0, -4.0)
	betaM := 0.28 * asing(dvt-40.0, 5.0, 5.0)
	alphaH := 0.128 * math.Exp(-(dvt-17.0)/18.0)
	betaH := 4.0 / (1.0 + math.Exp(-(dvt-40.0)/5.0))
	alphaN := -0.032 * asing(dvt-15.0, -5.0, -5.0)
	betaN := 0.5 * math.Exp(-(dvt-10.0)/40.0)
	dm := alphaM*(1.0-m) - betaM*m
	dh := alphaH*(1.0-h) - betaH*h
	dn := alphaN*(1.0-n) - betaN*n
	pInf := 1.0 / (1.0 + math.Exp(-(v+35.0)/10.0))
	tauP := 400.0 / (3.3*math.Exp((v+35.0)/20.0) + math.Exp(-(v+35.0)/20.0))
	dp := (pInf - p) / tauP
	mTInf := 1.0 / (1.0 + math.Exp(-(v+57.0)/6.2))
	sInf := 1.0 / (1.0 + math.Exp((v+81.0)/4.0))
	tauS := 30.0 + 200.0/(1.0+math.Exp((v+70.0)/5.0))
	ds := (sInf - sg) / tauS
	rInf := 1.0 / (1.0 + math.Exp((v+80.0)/10.0))
	tauR := 100.0 + 500.0/(math.Exp(-(v+70.0)/20.0)+math.Exp((v+70.0)/20.0))
	dr := (rInf - r) / tauR
	iNa := s.GNa * m * m * m * h * (v - s.ENa)
	iK := s.GK * n * n * n * n * (v - s.EK)
	iM := s.GM * p * (v - s.EK)
	iT := s.GT * mTInf * mTInf * sg * (v - s.ECa)
	iH := s.GH * r * (v - s.EH)
	iL := s.GL * (v - s.EL)
	dv := (-iNa - iK - iM - iT - iH - iL + current) / s.CM
	return [7]float64{dv, dm, dh, dn, dp, ds, dr}
}

// rk4Substep returns one classical RK4 increment of (V, m, h, n, p, s, r).
func (s *SSTNeuronState) rk4Substep(st [7]float64, current float64) [7]float64 {
	dt := s.Dt
	k1 := s.derivatives(st[0], st[1], st[2], st[3], st[4], st[5], st[6], current)
	var a [7]float64
	for i := 0; i < 7; i++ {
		a[i] = st[i] + 0.5*dt*k1[i]
	}
	k2 := s.derivatives(a[0], a[1], a[2], a[3], a[4], a[5], a[6], current)
	for i := 0; i < 7; i++ {
		a[i] = st[i] + 0.5*dt*k2[i]
	}
	k3 := s.derivatives(a[0], a[1], a[2], a[3], a[4], a[5], a[6], current)
	for i := 0; i < 7; i++ {
		a[i] = st[i] + dt*k3[i]
	}
	k4 := s.derivatives(a[0], a[1], a[2], a[3], a[4], a[5], a[6], current)
	var out [7]float64
	for i := 0; i < 7; i++ {
		out[i] = st[i] + dt*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])/6.0
	}
	return out
}

// Step advances the neuron by one 4*dt step and reports an upward threshold crossing.
func (s *SSTNeuronState) Step(iExt float64) int {
	vPrev := s.V
	st := [7]float64{s.V, s.M, s.H, s.N, s.P, s.S, s.R}
	for i := 0; i < 4; i++ {
		st = s.rk4Substep(st, iExt)
	}
	s.V, s.M, s.H, s.N, s.P, s.S, s.R = st[0], st[1], st[2], st[3], st[4], st[5], st[6]
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateSSTNeuron runs the neuron for nSteps and returns (trace, spikes).
func SimulateSSTNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSSTNeuron()
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
