// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for connor_stevens

package services

import (
	"errors"
	"math"
)

// ConnorStevensNeuronState holds the six-state Connor-Stevens conductance model.
type ConnorStevensNeuronState struct {
	V          float64
	M          float64
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
	EA         float64
	EL         float64
	CM         float64
	Dt         float64
	VThreshold float64
}

// ConnorStevensNeuron is kept as a compatibility alias for generated callers.
type ConnorStevensNeuron = ConnorStevensNeuronState

// NewConnorStevensNeuron creates a Connor-Stevens neuron with default parameters.
func NewConnorStevensNeuron() *ConnorStevensNeuronState {
	return &ConnorStevensNeuronState{
		V: -68.0, M: 0.01, H: 0.99, N: 0.1, A: 0.5, B: 0.1,
		GNa: 120.0, GK: 20.0, GA: 47.7, GL: 0.3,
		ENa: 55.0, EK: -72.0, EA: -75.0, EL: -17.0,
		CM: 1.0, Dt: 0.01, VThreshold: 0.0,
	}
}

type connorStevensDeriv struct {
	v float64
	m float64
	h float64
	n float64
	a float64
	b float64
}

func connorFinite(xs ...float64) bool {
	for _, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

func connorSafeExp(x float64) (float64, bool) {
	if !connorFinite(x) || x > 700.0 {
		return 0, false
	}
	return math.Exp(x), true
}

func connorSafeRate(scale, shift, v, denom float64) (float64, bool) {
	delta := v + shift
	x := delta / denom
	if math.Abs(x) < 1e-9 {
		return scale * denom, true
	}
	e, ok := connorSafeExp(-x)
	if !ok {
		return 0, false
	}
	value := scale * delta / (1.0 - e)
	return value, connorFinite(value)
}

func (s *ConnorStevensNeuronState) validStatic() bool {
	return connorFinite(s.GNa, s.GK, s.GA, s.GL, s.ENa, s.EK, s.EA, s.EL, s.CM, s.Dt, s.VThreshold) &&
		s.GNa >= 0.0 && s.GK >= 0.0 && s.GA >= 0.0 && s.GL >= 0.0 && s.CM > 0.0 && s.Dt > 0.0
}

func (s *ConnorStevensNeuronState) validState(v, m, h, n, a, b float64) bool {
	return connorFinite(v, m, h, n, a, b) &&
		v >= -250.0 && v <= 250.0 &&
		m >= -0.05 && m <= 1.05 && h >= -0.05 && h <= 1.05 && n >= -0.05 && n <= 1.05 &&
		a >= -0.05 && a <= 1.5 && b >= -0.05 && b <= 1.05
}

func (s *ConnorStevensNeuronState) derivatives(state connorStevensDeriv, current float64) (connorStevensDeriv, bool) {
	am, ok := connorSafeRate(0.38, 29.7, state.v, 10.0)
	if !ok {
		return connorStevensDeriv{}, false
	}
	bmExp, ok := connorSafeExp(-(state.v + 54.7) / 18.0)
	if !ok {
		return connorStevensDeriv{}, false
	}
	ahExp, ok := connorSafeExp(-(state.v + 48.0) / 20.0)
	if !ok {
		return connorStevensDeriv{}, false
	}
	bhExp, ok := connorSafeExp(-(state.v + 18.0) / 10.0)
	if !ok {
		return connorStevensDeriv{}, false
	}
	an, ok := connorSafeRate(0.02, 45.7, state.v, 10.0)
	if !ok {
		return connorStevensDeriv{}, false
	}
	bnExp, ok := connorSafeExp(-(state.v + 55.7) / 80.0)
	if !ok {
		return connorStevensDeriv{}, false
	}
	aNum, ok := connorSafeExp((state.v + 94.22) / 31.84)
	if !ok {
		return connorStevensDeriv{}, false
	}
	aDen, ok := connorSafeExp((state.v + 1.17) / 28.93)
	if !ok {
		return connorStevensDeriv{}, false
	}
	tauAExp, ok := connorSafeExp((state.v + 55.96) / 20.12)
	if !ok {
		return connorStevensDeriv{}, false
	}
	bExp, ok := connorSafeExp((state.v + 53.3) / 14.54)
	if !ok {
		return connorStevensDeriv{}, false
	}
	tauBExp, ok := connorSafeExp((state.v + 50.0) / 16.027)
	if !ok {
		return connorStevensDeriv{}, false
	}

	bm := 15.2 * bmExp
	ah := 0.266 * ahExp
	bh := 3.8 / (1.0 + bhExp)
	bn := 0.25 * bnExp
	aBase := 0.0761 * aNum / (1.0 + aDen)
	if aBase < 0.0 || !connorFinite(aBase) {
		return connorStevensDeriv{}, false
	}
	aInf := math.Pow(aBase, 1.0/3.0)
	tauA := 0.3632 + 1.158/(1.0+tauAExp)
	bBase := 1.0 / (1.0 + bExp)
	bInf := math.Pow(bBase, 4.0)
	tauB := 1.24 + 2.678/(1.0+tauBExp)
	iNa := s.GNa * math.Pow(state.m, 3.0) * state.h * (state.v - s.ENa)
	iK := s.GK * math.Pow(state.n, 4.0) * (state.v - s.EK)
	iA := s.GA * math.Pow(state.a, 3.0) * state.b * (state.v - s.EA)
	iL := s.GL * (state.v - s.EL)
	deriv := connorStevensDeriv{
		v: (-iNa - iK - iA - iL + current) / s.CM,
		m: am*(1.0-state.m) - bm*state.m,
		h: ah*(1.0-state.h) - bh*state.h,
		n: an*(1.0-state.n) - bn*state.n,
		a: (aInf - state.a) / tauA,
		b: (bInf - state.b) / tauB,
	}
	return deriv, connorFinite(deriv.v, deriv.m, deriv.h, deriv.n, deriv.a, deriv.b)
}

func (s *ConnorStevensNeuronState) rk4Candidate(current float64) (connorStevensDeriv, bool) {
	if !s.validStatic() || !connorFinite(current) || !s.validState(s.V, s.M, s.H, s.N, s.A, s.B) {
		return connorStevensDeriv{}, false
	}
	state := connorStevensDeriv{v: s.V, m: s.M, h: s.H, n: s.N, a: s.A, b: s.B}
	substeps := int(1.0 / math.Max(s.Dt, 0.001))
	for i := 0; i < substeps; i++ {
		k1, ok := s.derivatives(state, current)
		if !ok {
			return connorStevensDeriv{}, false
		}
		k2, ok := s.derivatives(connorStevensDeriv{v: state.v + 0.5*s.Dt*k1.v, m: state.m + 0.5*s.Dt*k1.m, h: state.h + 0.5*s.Dt*k1.h, n: state.n + 0.5*s.Dt*k1.n, a: state.a + 0.5*s.Dt*k1.a, b: state.b + 0.5*s.Dt*k1.b}, current)
		if !ok {
			return connorStevensDeriv{}, false
		}
		k3, ok := s.derivatives(connorStevensDeriv{v: state.v + 0.5*s.Dt*k2.v, m: state.m + 0.5*s.Dt*k2.m, h: state.h + 0.5*s.Dt*k2.h, n: state.n + 0.5*s.Dt*k2.n, a: state.a + 0.5*s.Dt*k2.a, b: state.b + 0.5*s.Dt*k2.b}, current)
		if !ok {
			return connorStevensDeriv{}, false
		}
		k4, ok := s.derivatives(connorStevensDeriv{v: state.v + s.Dt*k3.v, m: state.m + s.Dt*k3.m, h: state.h + s.Dt*k3.h, n: state.n + s.Dt*k3.n, a: state.a + s.Dt*k3.a, b: state.b + s.Dt*k3.b}, current)
		if !ok {
			return connorStevensDeriv{}, false
		}
		state = connorStevensDeriv{
			v: state.v + s.Dt*(k1.v+2.0*k2.v+2.0*k3.v+k4.v)/6.0,
			m: state.m + s.Dt*(k1.m+2.0*k2.m+2.0*k3.m+k4.m)/6.0,
			h: state.h + s.Dt*(k1.h+2.0*k2.h+2.0*k3.h+k4.h)/6.0,
			n: state.n + s.Dt*(k1.n+2.0*k2.n+2.0*k3.n+k4.n)/6.0,
			a: state.a + s.Dt*(k1.a+2.0*k2.a+2.0*k3.a+k4.a)/6.0,
			b: state.b + s.Dt*(k1.b+2.0*k2.b+2.0*k3.b+k4.b)/6.0,
		}
		if !s.validState(state.v, state.m, state.h, state.n, state.a, state.b) {
			return connorStevensDeriv{}, false
		}
	}
	return state, true
}

// Step advances the neuron by one macro-step using candidate-first RK4 sub-steps.
func (s *ConnorStevensNeuronState) Step(iExt float64) (int, error) {
	vPrev := s.V
	candidate, ok := s.rk4Candidate(iExt)
	if !ok {
		return 0, errors.New("invalid Connor-Stevens state, parameters, current, or candidate")
	}
	s.V, s.M, s.H, s.N, s.A, s.B = candidate.v, candidate.m, candidate.h, candidate.n, candidate.a, candidate.b
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateConnorStevensNeuron runs the neuron for n steps and records voltage.
func SimulateConnorStevensNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewConnorStevensNeuron()
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
