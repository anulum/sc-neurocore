// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for Sherman-Rinzel-Keizer beta-cell burster

package services

import "math"

const srkTauN = 9.09

// ShermanRinzelKeizerNeuronState holds the neuron state.
type ShermanRinzelKeizerNeuronState struct {
	V          float64
	N          float64
	S          float64
	GCa        float64
	GK         float64
	GS         float64
	ECa        float64
	EK         float64
	TauS       float64
	Dt         float64
	VThreshold float64
}

// NewShermanRinzelKeizerNeuron creates a neuron with the documented defaults.
func NewShermanRinzelKeizerNeuron() *ShermanRinzelKeizerNeuronState {
	return &ShermanRinzelKeizerNeuronState{V: -50.0, N: 0.1, S: 0.1, GCa: 3.6, GK: 10.0, GS: 4.0, ECa: 25.0, EK: -75.0, TauS: 5000.0, Dt: 0.5, VThreshold: -20.0}
}

func srkFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func srkGate(value float64) bool { return srkFinite(value) && value >= 0.0 && value <= 1.0 }

func srkSigmoid(arg float64) float64 {
	if arg > 80.0 {
		arg = 80.0
	} else if arg < -80.0 {
		arg = -80.0
	}
	return 1.0 / (1.0 + math.Exp(-arg))
}

func (s *ShermanRinzelKeizerNeuronState) valid() bool {
	return srkFinite(s.V) && s.V >= -200.0 && s.V <= 200.0 &&
		srkGate(s.N) && srkGate(s.S) &&
		srkFinite(s.GCa) && s.GCa > 0.0 &&
		srkFinite(s.GK) && s.GK > 0.0 &&
		srkFinite(s.GS) && s.GS >= 0.0 &&
		srkFinite(s.ECa) && srkFinite(s.EK) &&
		srkFinite(s.TauS) && s.TauS > 0.0 &&
		srkFinite(s.Dt) && s.Dt > 0.0 &&
		srkFinite(s.VThreshold)
}

func (s *ShermanRinzelKeizerNeuronState) derivatives(v float64, nGate float64, sGate float64, iExt float64) (float64, float64, float64, bool) {
	if !(srkFinite(v) && srkFinite(nGate) && srkFinite(sGate) && srkFinite(iExt)) {
		return 0, 0, 0, false
	}
	mInf := srkSigmoid((v + 20.0) / 12.0)
	nInf := srkSigmoid((v + 16.0) / 5.0)
	sInf := srkSigmoid((v + 35.0) / 10.0)
	iCa := s.GCa * mInf * (v - s.ECa)
	iK := s.GK * nGate * (v - s.EK)
	iS := s.GS * sGate * (v - s.EK)
	dv := -iCa - iK - iS + iExt
	dn := (nInf - nGate) / srkTauN
	ds := (sInf - sGate) / s.TauS
	return dv, dn, ds, srkFinite(dv) && srkFinite(dn) && srkFinite(ds)
}

func (s *ShermanRinzelKeizerNeuronState) rk4Candidate(iExt float64) (float64, float64, float64, bool) {
	halfDt := 0.5 * s.Dt
	k1v, k1n, k1s, ok := s.derivatives(s.V, s.N, s.S, iExt)
	if !ok {
		return 0, 0, 0, false
	}
	k2v, k2n, k2s, ok := s.derivatives(s.V+halfDt*k1v, s.N+halfDt*k1n, s.S+halfDt*k1s, iExt)
	if !ok {
		return 0, 0, 0, false
	}
	k3v, k3n, k3s, ok := s.derivatives(s.V+halfDt*k2v, s.N+halfDt*k2n, s.S+halfDt*k2s, iExt)
	if !ok {
		return 0, 0, 0, false
	}
	k4v, k4n, k4s, ok := s.derivatives(s.V+s.Dt*k3v, s.N+s.Dt*k3n, s.S+s.Dt*k3s, iExt)
	if !ok {
		return 0, 0, 0, false
	}
	nextV := s.V + s.Dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
	nextN := s.N + s.Dt*(k1n+2.0*k2n+2.0*k3n+k4n)/6.0
	nextS := s.S + s.Dt*(k1s+2.0*k2s+2.0*k3s+k4s)/6.0
	ok = srkFinite(nextV) && nextV >= -200.0 && nextV <= 200.0 && srkGate(nextN) && srkGate(nextS)
	return nextV, nextN, nextS, ok
}

// Step advances the neuron by one RK4 timestep and returns the threshold crossing indicator.
func (s *ShermanRinzelKeizerNeuronState) Step(iExt float64) int {
	if !s.valid() || !srkFinite(iExt) {
		return 0
	}
	vPrev := s.V
	nextV, nextN, nextS, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0
	}
	s.V = nextV
	s.N = nextN
	s.S = nextS
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateShermanRinzelKeizerNeuron runs the neuron for n steps.
func SimulateShermanRinzelKeizerNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewShermanRinzelKeizerNeuron()
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
