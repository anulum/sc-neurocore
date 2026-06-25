// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go candidate-first RK4 service for hill_tononi

package services

import (
	"math"
)

// HillTononiNeuronState holds the neuron state
type HillTononiNeuronState struct {
	V          float64
	HNa        float64
	NK         float64
	MH         float64
	HT         float64
	NaI        float64
	GNa        float64
	GK         float64
	GH         float64
	GT         float64
	GKna       float64
	GL         float64
	ENa        float64
	EK         float64
	EH         float64
	ECa        float64
	EL         float64
	NaPumpMax  float64
	NaEq       float64
	Dt         float64
	VThreshold float64
}

// NewHillTononiNeuron creates a new HillTononiNeuron neuron with default parameters
func NewHillTononiNeuron() *HillTononiNeuronState {
	return &HillTononiNeuronState{
		V:          -65.0,
		HNa:        0.6,
		NK:         0.3,
		MH:         0.0,
		HT:         0.9,
		NaI:        5.0,
		GNa:        50.0,
		GK:         5.0,
		GH:         1.0,
		GT:         3.0,
		GKna:       1.33,
		GL:         0.02,
		ENa:        50.0,
		EK:         -90.0,
		EH:         -43.0,
		ECa:        120.0,
		EL:         -70.0,
		NaPumpMax:  20.0,
		NaEq:       9.5,
		Dt:         0.05,
		VThreshold: -20.0,
	}
}

// derivatives returns (dV, dHNa, dNK, dMH, dHT, dNaI) evaluated from one
// consistent state. The conductance powers use explicit multiplication and the
// I_KNa Hill exponent 3.5 is evaluated as b*b*b*sqrt(b) (an IEEE-754 exact
// decomposition) so the Python, Rust, Julia, and Mojo backends reproduce the
// trajectory bit-for-bit rather than depending on math.Pow.
func (s *HillTononiNeuronState) derivatives(v, hNa, nK, mH, hT, naI, current float64) (float64, float64, float64, float64, float64, float64) {
	mNaInf := 1.0 / (1.0 + math.Exp(-(v+38.0)/10.0))
	hNaInf := 1.0 / (1.0 + math.Exp((v+43.0)/6.0))
	nKInf := 1.0 / (1.0 + math.Exp(-(v+27.0)/11.5))
	mHInf := 1.0 / (1.0 + math.Exp((v+75.0)/5.5))
	mTInf := 1.0 / (1.0 + math.Exp(-(v+59.0)/6.2))
	hTInf := 1.0 / (1.0 + math.Exp((v+83.0)/4.0))
	hillBase := 38.7 / math.Max(naI, 0.01)
	wKna := 0.37 / (1.0 + hillBase*hillBase*hillBase*math.Sqrt(hillBase))
	tauHNa := math.Max(1.0+10.0/(1.0+math.Exp((v+40.0)/10.0)), 0.1)
	zN := (v + 50.0) / 25.0
	tauNK := math.Max(5.0+47.0*math.Exp(-(zN*zN)), 0.1)
	tauMH := math.Max(20.0+1000.0/(math.Exp((v+71.5)/14.2)+math.Exp(-(v+89.0)/11.6)), 1.0)
	tauHT := 10.0
	if v < -81.0 {
		tauHT = math.Max(30.8+211.4*math.Exp((v+115.2)/5.0)/(1.0+math.Exp((v+86.0)/3.2)), 0.1)
	}
	dHNa := (hNaInf - hNa) / tauHNa
	dNK := (nKInf - nK) / tauNK
	dMH := (mHInf - mH) / tauMH
	dHT := (hTInf - hT) / tauHT
	iNa := s.GNa * mNaInf * mNaInf * mNaInf * hNa * (v - s.ENa)
	iK := s.GK * nK * nK * nK * nK * (v - s.EK)
	iH := s.GH * mH * (v - s.EH)
	iT := s.GT * mTInf * mTInf * hT * (v - s.ECa)
	iKna := s.GKna * wKna * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dV := -iNa - iK - iH - iT - iKna - iL + current
	dNaI := -0.001*iNa - s.NaPumpMax*(naI/(naI+s.NaEq))
	return dV, dHNa, dNK, dMH, dHT, dNaI
}

// rk4Substep returns one classical RK4 increment of the six-state vector over Dt.
func (s *HillTononiNeuronState) rk4Substep(v, hNa, nK, mH, hT, naI, current float64) (float64, float64, float64, float64, float64, float64) {
	dt := s.Dt
	k1v, k1hNa, k1nK, k1mH, k1hT, k1naI := s.derivatives(v, hNa, nK, mH, hT, naI, current)
	k2v, k2hNa, k2nK, k2mH, k2hT, k2naI := s.derivatives(v+0.5*dt*k1v, hNa+0.5*dt*k1hNa, nK+0.5*dt*k1nK, mH+0.5*dt*k1mH, hT+0.5*dt*k1hT, naI+0.5*dt*k1naI, current)
	k3v, k3hNa, k3nK, k3mH, k3hT, k3naI := s.derivatives(v+0.5*dt*k2v, hNa+0.5*dt*k2hNa, nK+0.5*dt*k2nK, mH+0.5*dt*k2mH, hT+0.5*dt*k2hT, naI+0.5*dt*k2naI, current)
	k4v, k4hNa, k4nK, k4mH, k4hT, k4naI := s.derivatives(v+dt*k3v, hNa+dt*k3hNa, nK+dt*k3nK, mH+dt*k3mH, hT+dt*k3hT, naI+dt*k3naI, current)
	nextV := v + dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
	nextHNa := hNa + dt*(k1hNa+2.0*k2hNa+2.0*k3hNa+k4hNa)/6.0
	nextNK := nK + dt*(k1nK+2.0*k2nK+2.0*k3nK+k4nK)/6.0
	nextMH := mH + dt*(k1mH+2.0*k2mH+2.0*k3mH+k4mH)/6.0
	nextHT := hT + dt*(k1hT+2.0*k2hT+2.0*k3hT+k4hT)/6.0
	nextNaI := naI + dt*(k1naI+2.0*k2naI+2.0*k3naI+k4naI)/6.0
	return nextV, nextHNa, nextNK, nextMH, nextHT, nextNaI
}

// Step advances the neuron by one RK4 timestep and reports a threshold crossing.
func (s *HillTononiNeuronState) Step(iExt float64) int {
	vPrev := s.V
	nextV, nextHNa, nextNK, nextMH, nextHT, nextNaI := s.rk4Substep(s.V, s.HNa, s.NK, s.MH, s.HT, s.NaI, iExt)
	s.V = nextV
	s.HNa = nextHNa
	s.NK = nextNK
	s.MH = nextMH
	s.HT = nextHT
	s.NaI = math.Max(nextNaI, 0.0)
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateHillTononiNeuron runs the neuron for n steps
func SimulateHillTononiNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewHillTononiNeuron()
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
