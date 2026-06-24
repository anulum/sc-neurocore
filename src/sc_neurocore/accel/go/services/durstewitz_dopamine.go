// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go candidate-first RK4 service for durstewitz_dopamine

package services

import (
	"math"
)

// DurstewitzDopamineNeuronState holds the neuron state
type DurstewitzDopamineNeuronState struct {
	V          float64
	HNa        float64
	NK         float64
	GNa        float64
	GK         float64
	GNmda      float64
	GL         float64
	ENa        float64
	EK         float64
	ENmda      float64
	EL         float64
	Mg         float64
	D1Level    float64
	GNmdaScale float64
	GKScale    float64
	VShiftNa   float64
	Dt         float64
	VThreshold float64
}

// NewDurstewitzDopamineNeuron creates a new DurstewitzDopamineNeuron neuron with default parameters
func NewDurstewitzDopamineNeuron() *DurstewitzDopamineNeuronState {
	return &DurstewitzDopamineNeuronState{
		V:          -65.0,
		HNa:        0.7,
		NK:         0.2,
		GNa:        45.0,
		GK:         18.0,
		GNmda:      0.5,
		GL:         0.02,
		ENa:        55.0,
		EK:         -80.0,
		ENmda:      0.0,
		EL:         -65.0,
		Mg:         1.0,
		D1Level:    0.0,
		GNmdaScale: 2.5,
		GKScale:    1.5,
		VShiftNa:   -5.0,
		Dt:         0.05,
		VThreshold: -20.0,
	}
}

// derivatives returns (dV, dHNa, dNK) evaluated from one consistent state. The
// sodium activation m_inf is instantaneous; the conductance powers use explicit
// multiplication and the Mg2+ block keeps the Mg/3.57*exp operand order so the
// Python, Rust, Julia, and Mojo backends reproduce the trajectory bit-for-bit.
func (s *DurstewitzDopamineNeuronState) derivatives(v float64, hNa float64, nK float64, current float64) (float64, float64, float64) {
	vSh := s.D1Level * s.VShiftNa
	mNaInf := 1.0 / (1.0 + math.Exp(-(v+30.0+vSh)/9.5))
	hNaInf := 1.0 / (1.0 + math.Exp((v+53.0)/7.0))
	nKInf := 1.0 / (1.0 + math.Exp(-(v+30.0)/10.0))
	tauH := 0.5 + 14.0/(1.0+math.Exp((v+50.0)/12.0))
	tauN := 1.0 + 11.0/(1.0+math.Exp((v+40.0)/10.0))
	dHNa := (hNaInf - hNa) / tauH
	dNK := (nKInf - nK) / tauN
	mgBlock := 1.0 / (1.0 + s.Mg/3.57*math.Exp(-0.062*v))
	nmdaG := s.GNmda * (1.0 + s.D1Level*(s.GNmdaScale-1.0))
	kG := s.GK * (1.0 + s.D1Level*(s.GKScale-1.0))
	iNa := s.GNa * mNaInf * mNaInf * mNaInf * hNa * (v - s.ENa)
	iK := kG * nK * nK * nK * nK * (v - s.EK)
	iNmda := nmdaG * mgBlock * (v - s.ENmda)
	iL := s.GL * (v - s.EL)
	dV := -iNa - iK - iNmda - iL + current
	return dV, dHNa, dNK
}

// rk4Substep returns one classical RK4 increment of (V, HNa, NK) over Dt.
func (s *DurstewitzDopamineNeuronState) rk4Substep(v float64, hNa float64, nK float64, current float64) (float64, float64, float64) {
	dt := s.Dt
	k1v, k1h, k1n := s.derivatives(v, hNa, nK, current)
	k2v, k2h, k2n := s.derivatives(v+0.5*dt*k1v, hNa+0.5*dt*k1h, nK+0.5*dt*k1n, current)
	k3v, k3h, k3n := s.derivatives(v+0.5*dt*k2v, hNa+0.5*dt*k2h, nK+0.5*dt*k2n, current)
	k4v, k4h, k4n := s.derivatives(v+dt*k3v, hNa+dt*k3h, nK+dt*k3n, current)
	nextV := v + dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
	nextH := hNa + dt*(k1h+2.0*k2h+2.0*k3h+k4h)/6.0
	nextN := nK + dt*(k1n+2.0*k2n+2.0*k3n+k4n)/6.0
	return nextV, nextH, nextN
}

// Step advances the neuron by one RK4 timestep and reports a threshold crossing.
func (s *DurstewitzDopamineNeuronState) Step(iExt float64) int {
	vPrev := s.V
	nextV, nextH, nextN := s.rk4Substep(s.V, s.HNa, s.NK, iExt)
	s.V = nextV
	s.HNa = nextH
	s.NK = nextN
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateDurstewitzDopamineNeuron runs the neuron for n steps
func SimulateDurstewitzDopamineNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewDurstewitzDopamineNeuron()
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
