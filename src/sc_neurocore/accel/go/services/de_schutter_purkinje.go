// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go candidate-first RK4 service for de_schutter_purkinje

package services

import "math"

const deSchutterSubsteps = 5

// DeSchutterPurkinjeNeuronState holds the neuron state.
type DeSchutterPurkinjeNeuronState struct {
	V          float64
	HNa        float64
	NK         float64
	MCap       float64
	HCap       float64
	QKca       float64
	Ca         float64
	GNa        float64
	GK         float64
	GCap       float64
	GKca       float64
	GL         float64
	ENa        float64
	EK         float64
	ECa        float64
	EL         float64
	CaDecay    float64
	FCa        float64
	Dt         float64
	VThreshold float64
}

// NewDeSchutterPurkinjeNeuron creates a new DeSchutterPurkinjeNeuron neuron with default parameters.
func NewDeSchutterPurkinjeNeuron() *DeSchutterPurkinjeNeuronState {
	return &DeSchutterPurkinjeNeuronState{
		V:          -68.0,
		HNa:        0.8,
		NK:         0.1,
		MCap:       0.0,
		HCap:       0.9,
		QKca:       0.0,
		Ca:         0.0001,
		GNa:        125.0,
		GK:         10.0,
		GCap:       45.0,
		GKca:       35.0,
		GL:         0.5,
		ENa:        45.0,
		EK:         -85.0,
		ECa:        135.0,
		EL:         -68.0,
		CaDecay:    0.02,
		FCa:        0.00024,
		Dt:         0.01,
		VThreshold: -20.0,
	}
}

func dspFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *DeSchutterPurkinjeNeuronState) valid() bool {
	return dspFinite(s.V, s.HNa, s.NK, s.MCap, s.HCap, s.QKca, s.Ca, s.GNa, s.GK, s.GCap, s.GKca, s.GL, s.ENa, s.EK, s.ECa, s.EL, s.CaDecay, s.FCa, s.Dt, s.VThreshold) &&
		s.Ca >= 0.0 && s.GNa >= 0.0 && s.GK >= 0.0 && s.GCap >= 0.0 && s.GKca >= 0.0 &&
		s.GL >= 0.0 && s.CaDecay >= 0.0 && s.FCa >= 0.0 && s.Dt > 0.0
}

func (s *DeSchutterPurkinjeNeuronState) derivatives(v, hNa, nK, mCap, hCap, qKca, ca, current float64) (float64, float64, float64, float64, float64, float64, float64) {
	caEff := math.Max(ca, 0.0)
	mNaInf := 1.0 / (1.0 + math.Exp(-(v+35.0)/7.5))
	hNaInf := 1.0 / (1.0 + math.Exp((v+55.0)/7.0))
	nKInf := 1.0 / (1.0 + math.Exp(-(v+30.0)/15.0))
	mCapInf := 1.0 / (1.0 + math.Exp(-(v+19.0)/5.5))
	hCapInf := 1.0 / (1.0 + math.Exp((v+48.0)/7.0))
	qKcaInf := caEff / (caEff + 0.0002)
	tauHNa := 0.5 + 14.0/(1.0+math.Exp((v+40.0)/12.0))
	tauNK := 1.0 + 11.0/(1.0+math.Exp((v+15.0)/8.0))
	dHNa := (hNaInf - hNa) / tauHNa
	dNK := (nKInf - nK) / tauNK
	dMCap := (mCapInf - mCap) / 0.3
	dHCap := (hCapInf - hCap) / 45.0
	dQKca := qKcaInf - qKca
	iNa := s.GNa * mNaInf * mNaInf * mNaInf * hNa * (v - s.ENa)
	iK := s.GK * nK * nK * nK * nK * (v - s.EK)
	iCap := s.GCap * mCap * mCap * hCap * (v - s.ECa)
	iKca := s.GKca * qKca * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dV := -iNa - iK - iCap - iKca - iL + current
	dCa := -s.FCa*iCap - s.CaDecay*caEff
	return dV, dHNa, dNK, dMCap, dHCap, dQKca, dCa
}

func (s *DeSchutterPurkinjeNeuronState) rk4Substep(v, hNa, nK, mCap, hCap, qKca, ca, current float64) (float64, float64, float64, float64, float64, float64, float64) {
	dt := s.Dt
	k1v, k1h, k1n, k1m, k1hc, k1q, k1ca := s.derivatives(v, hNa, nK, mCap, hCap, qKca, ca, current)
	k2v, k2h, k2n, k2m, k2hc, k2q, k2ca := s.derivatives(v+0.5*dt*k1v, hNa+0.5*dt*k1h, nK+0.5*dt*k1n, mCap+0.5*dt*k1m, hCap+0.5*dt*k1hc, qKca+0.5*dt*k1q, ca+0.5*dt*k1ca, current)
	k3v, k3h, k3n, k3m, k3hc, k3q, k3ca := s.derivatives(v+0.5*dt*k2v, hNa+0.5*dt*k2h, nK+0.5*dt*k2n, mCap+0.5*dt*k2m, hCap+0.5*dt*k2hc, qKca+0.5*dt*k2q, ca+0.5*dt*k2ca, current)
	k4v, k4h, k4n, k4m, k4hc, k4q, k4ca := s.derivatives(v+dt*k3v, hNa+dt*k3h, nK+dt*k3n, mCap+dt*k3m, hCap+dt*k3hc, qKca+dt*k3q, ca+dt*k3ca, current)
	nextV := v + dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
	nextHNa := hNa + dt*(k1h+2.0*k2h+2.0*k3h+k4h)/6.0
	nextNK := nK + dt*(k1n+2.0*k2n+2.0*k3n+k4n)/6.0
	nextMCap := mCap + dt*(k1m+2.0*k2m+2.0*k3m+k4m)/6.0
	nextHCap := hCap + dt*(k1hc+2.0*k2hc+2.0*k3hc+k4hc)/6.0
	nextQKca := qKca + dt*(k1q+2.0*k2q+2.0*k3q+k4q)/6.0
	nextCa := ca + dt*(k1ca+2.0*k2ca+2.0*k3ca+k4ca)/6.0
	return nextV, nextHNa, nextNK, nextMCap, nextHCap, nextQKca, math.Max(nextCa, 0.0)
}

// Step advances the neuron by one candidate-first RK4 timestep.
func (s *DeSchutterPurkinjeNeuronState) Step(iExt float64) int {
	if !dspFinite(iExt) || !s.valid() {
		return 0
	}
	vPrev := s.V
	v, hNa, nK, mCap, hCap, qKca, ca := s.V, s.HNa, s.NK, s.MCap, s.HCap, s.QKca, s.Ca
	for i := 0; i < deSchutterSubsteps; i++ {
		v, hNa, nK, mCap, hCap, qKca, ca = s.rk4Substep(v, hNa, nK, mCap, hCap, qKca, ca, iExt)
		if !dspFinite(v, hNa, nK, mCap, hCap, qKca, ca) {
			return 0
		}
	}
	s.V = v
	s.HNa = hNa
	s.NK = nK
	s.MCap = mCap
	s.HCap = hCap
	s.QKca = qKca
	s.Ca = ca
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateDeSchutterPurkinjeNeuron runs the neuron for n steps.
func SimulateDeSchutterPurkinjeNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDeSchutterPurkinjeNeuron()
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
