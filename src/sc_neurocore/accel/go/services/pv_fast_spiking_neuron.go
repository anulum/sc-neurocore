// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go RK4 kernel for the PV+ fast-spiking neuron

package services

import (
	"math"
)

// PVFastSpikingNeuronState holds the four-state Wang-Buzsáki + Kv3.1 neuron.
type PVFastSpikingNeuronState struct {
	V          float64
	H          float64
	N          float64
	P          float64
	GNa        float64
	GK         float64
	GKv3       float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	Dt         float64
	VThreshold float64
}

// NewPVFastSpikingNeuron creates a PV+ fast-spiking neuron with the published defaults.
func NewPVFastSpikingNeuron() *PVFastSpikingNeuronState {
	return &PVFastSpikingNeuronState{
		V: -65.0, H: 0.8, N: 0.1, P: 0.0,
		GNa: 35.0, GK: 9.0, GKv3: 5.0, GL: 0.1,
		ENa: 55.0, EK: -90.0, EL: -65.0, CM: 1.0,
		Phi: 5.0, Dt: 0.01, VThreshold: -20.0,
	}
}

// safeRate returns a(v+vhalf)/(1-exp(-(v+vhalf)/k)) with the closed-form L'Hôpital
// limit (a*k, passed as fallback) within 1e-7 of the removable singularity.
func safePVFSRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

// derivatives returns [dV, dh, dn, dp] at one consistent state.
func (s *PVFastSpikingNeuronState) derivatives(v, h, n, p, current float64) [4]float64 {
	am := safePVFSRate(0.1, 35.0, v, 10.0, 1.0)
	bm := 4.0 * math.Exp(-(v+60.0)/18.0)
	mInf := am / (am + bm)
	ah := 0.07 * math.Exp(-(v+58.0)/20.0)
	bh := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
	an := safePVFSRate(0.01, 34.0, v, 10.0, 0.1)
	bn := 0.125 * math.Exp(-(v+44.0)/80.0)
	pInf := 1.0 / (1.0 + math.Exp(-(v+10.0)/10.0))
	dh := s.Phi * (ah*(1.0-h) - bh*h)
	dn := s.Phi * (an*(1.0-n) - bn*n)
	dp := s.Phi * (pInf - p)
	iNa := s.GNa * mInf * mInf * mInf * h * (v - s.ENa)
	iK := s.GK * n * n * n * n * (v - s.EK)
	iKv3 := s.GKv3 * p * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dv := (-iNa - iK - iKv3 - iL + current) / s.CM
	return [4]float64{dv, dh, dn, dp}
}

// rk4Substep returns one classical RK4 increment of (V, h, n, p).
func (s *PVFastSpikingNeuronState) rk4Substep(st [4]float64, current float64) [4]float64 {
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

// Step advances the neuron by one 0.5 ms step and reports an upward threshold crossing.
func (s *PVFastSpikingNeuronState) Step(iExt float64) int {
	vPrev := s.V
	nSub := int(0.5 / math.Max(s.Dt, 0.001))
	if nSub < 1 {
		nSub = 1
	}
	st := [4]float64{s.V, s.H, s.N, s.P}
	for i := 0; i < nSub; i++ {
		st = s.rk4Substep(st, iExt)
	}
	s.V, s.H, s.N, s.P = st[0], st[1], st[2], st[3]
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulatePVFastSpikingNeuron runs the neuron for nSteps and returns (trace, spikes).
func SimulatePVFastSpikingNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPVFastSpikingNeuron()
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
