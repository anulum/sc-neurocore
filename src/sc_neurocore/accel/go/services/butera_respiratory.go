// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for butera_respiratory

package services

import (
	"errors"
	"math"
)

// ButeraRespiratoryNeuronState holds the Butera Model I respiratory state.
type ButeraRespiratoryNeuronState struct {
	V           float64
	N           float64
	HNap        float64
	GNa         float64
	GNap        float64
	GK          float64
	GL          float64
	Capacitance float64
	ENa         float64
	EK          float64
	EL          float64
	GTonic      float64
	ESyn        float64
	TauH        float64
	Dt          float64
	VThreshold  float64
}

// NewButeraRespiratoryNeuron creates a new ButeraRespiratoryNeuron neuron with default parameters.
func NewButeraRespiratoryNeuron() *ButeraRespiratoryNeuronState {
	return &ButeraRespiratoryNeuronState{V: -50.0, N: 0.01, HNap: 0.5, GNa: 28.0, GNap: 2.8, GK: 11.2, GL: 2.8, Capacitance: 21.0, ENa: 50.0, EK: -85.0, EL: -65.0, GTonic: 0.0, ESyn: 0.0, TauH: 10000.0, Dt: 0.1, VThreshold: -20.0}
}

type buteraDeriv struct{ v, n, hNap float64 }

func buteraFinite(xs ...float64) bool {
	for _, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

func (s *ButeraRespiratoryNeuronState) validStatic() bool {
	return buteraFinite(s.GNa, s.GNap, s.GK, s.GL, s.Capacitance, s.ENa, s.EK, s.EL, s.GTonic, s.ESyn, s.TauH, s.Dt, s.VThreshold) && s.GNa >= 0 && s.GNap >= 0 && s.GK >= 0 && s.GL >= 0 && s.Capacitance > 0 && s.GTonic >= 0 && s.TauH > 0 && s.Dt > 0
}

func buteraValidState(v, n, hNap float64) bool {
	return buteraFinite(v, n, hNap) && v >= -200.0 && v <= 100.0 && n >= -0.05 && n <= 1.05 && hNap >= -0.05 && hNap <= 1.05
}

func buteraRates(v, tauHBase float64) (float64, float64, float64, float64, float64, float64, bool) {
	mNa := 1.0 / (1.0 + math.Exp(-(v+34.0)/5.0))
	mNap := 1.0 / (1.0 + math.Exp(-(v+40.0)/6.0))
	hNapInf := 1.0 / (1.0 + math.Exp((v+48.0)/6.0))
	nInf := 1.0 / (1.0 + math.Exp(-(v+29.0)/4.0))
	tauN := math.Max(10.0/math.Max(math.Cosh((v+29.0)/8.0), 1e-12), 0.01)
	tauH := math.Max(tauHBase/math.Max(math.Cosh((v+48.0)/12.0), 1e-12), 0.1)
	return mNa, mNap, hNapInf, nInf, tauN, tauH, buteraFinite(mNa, mNap, hNapInf, nInf, tauN, tauH) && tauN > 0 && tauH > 0
}

func (s *ButeraRespiratoryNeuronState) derivatives(state buteraDeriv, current float64) (buteraDeriv, bool) {
	if !buteraFinite(state.v, state.n, state.hNap, current) {
		return buteraDeriv{}, false
	}
	state.v = math.Max(-200.0, math.Min(100.0, state.v))
	state.n = math.Max(0.0, math.Min(1.0, state.n))
	state.hNap = math.Max(0.0, math.Min(1.0, state.hNap))
	mNa, mNap, hNapInf, nInf, tauN, tauH, ok := buteraRates(state.v, s.TauH)
	if !ok {
		return buteraDeriv{}, false
	}
	iNa := s.GNa * math.Pow(mNa, 3) * (1.0 - state.n) * (state.v - s.ENa)
	iNap := s.GNap * mNap * state.hNap * (state.v - s.ENa)
	iK := s.GK * math.Pow(state.n, 4) * (state.v - s.EK)
	iL := s.GL * (state.v - s.EL)
	iTonic := s.GTonic * (state.v - s.ESyn)
	deriv := buteraDeriv{v: (-iNa - iNap - iK - iL - iTonic + current) / s.Capacitance, n: (nInf - state.n) / tauN, hNap: (hNapInf - state.hNap) / tauH}
	return deriv, buteraFinite(deriv.v, deriv.n, deriv.hNap)
}

func (s *ButeraRespiratoryNeuronState) rk4Candidate(current float64) (buteraDeriv, bool) {
	if !s.validStatic() || !buteraFinite(current) || !buteraValidState(s.V, s.N, s.HNap) {
		return buteraDeriv{}, false
	}
	state := buteraDeriv{v: s.V, n: s.N, hNap: s.HNap}
	k1, ok := s.derivatives(state, current)
	if !ok {
		return buteraDeriv{}, false
	}
	k2, ok := s.derivatives(buteraDeriv{v: state.v + 0.5*s.Dt*k1.v, n: state.n + 0.5*s.Dt*k1.n, hNap: state.hNap + 0.5*s.Dt*k1.hNap}, current)
	if !ok {
		return buteraDeriv{}, false
	}
	k3, ok := s.derivatives(buteraDeriv{v: state.v + 0.5*s.Dt*k2.v, n: state.n + 0.5*s.Dt*k2.n, hNap: state.hNap + 0.5*s.Dt*k2.hNap}, current)
	if !ok {
		return buteraDeriv{}, false
	}
	k4, ok := s.derivatives(buteraDeriv{v: state.v + s.Dt*k3.v, n: state.n + s.Dt*k3.n, hNap: state.hNap + s.Dt*k3.hNap}, current)
	if !ok {
		return buteraDeriv{}, false
	}
	candidate := buteraDeriv{v: state.v + s.Dt*(k1.v+2*k2.v+2*k3.v+k4.v)/6, n: state.n + s.Dt*(k1.n+2*k2.n+2*k3.n+k4.n)/6, hNap: state.hNap + s.Dt*(k1.hNap+2*k2.hNap+2*k3.hNap+k4.hNap)/6}
	if !buteraFinite(candidate.v, candidate.n, candidate.hNap) {
		return buteraDeriv{}, false
	}
	candidate.v = math.Max(-200.0, math.Min(100.0, candidate.v))
	candidate.n = math.Max(0.0, math.Min(1.0, candidate.n))
	candidate.hNap = math.Max(0.0, math.Min(1.0, candidate.hNap))
	return candidate, true
}

// Step advances the neuron by one timestep using candidate-first RK4.
func (s *ButeraRespiratoryNeuronState) Step(iExt float64) (int, error) {
	vPrev := s.V
	candidate, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0, errors.New("invalid Butera respiratory state, parameter, current, or candidate")
	}
	s.V, s.N, s.HNap = candidate.v, candidate.n, candidate.hNap
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateButeraRespiratoryNeuron runs the neuron for n steps.
func SimulateButeraRespiratoryNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewButeraRespiratoryNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = s.V
			continue
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
