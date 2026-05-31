// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for yamada

package services

import "math"

// YamadaNeuronState holds the neuron state.
type YamadaNeuronState struct {
	V          float64
	N          float64
	Q          float64
	GNa        float64
	GK         float64
	GQ         float64
	GL         float64
	ENa        float64
	EK         float64
	EQ         float64
	EL         float64
	TauQ       float64
	Dt         float64
	VThreshold float64
}

func yamadaSigmoid(x float64) float64 {
	if x >= 0.0 {
		z := math.Exp(-x)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(x)
	return z / (1.0 + z)
}

func yamadaTauN(v float64) float64 {
	x := (v + 40.0) / 12.0
	if !finite(x) {
		return math.NaN()
	}
	if x > 709.0 {
		return 1.0
	}
	return 1.0 + 7.5/(1.0+math.Exp(x))
}

type yamadaDerivative struct {
	v float64
	n float64
	q float64
}

// NewYamadaNeuron creates a new YamadaNeuron neuron with default parameters.
func NewYamadaNeuron() *YamadaNeuronState {
	return &YamadaNeuronState{V: -60.0, N: 0.1, Q: 0.0, GNa: 20.0, GK: 10.0, GQ: 5.0, GL: 0.5, ENa: 60.0, EK: -80.0, EQ: -80.0, EL: -60.0, TauQ: 300.0, Dt: 0.05, VThreshold: -20.0}
}

// Valid reports whether parameters and gates satisfy the Yamada scalar contract.
func (s YamadaNeuronState) Valid() bool {
	return finite(s.V) && finite(s.N) && s.N >= 0.0 && s.N <= 1.0 &&
		finite(s.Q) && s.Q >= 0.0 && s.Q <= 1.0 &&
		finite(s.GNa) && s.GNa >= 0.0 && finite(s.GK) && s.GK >= 0.0 &&
		finite(s.GQ) && s.GQ >= 0.0 && finite(s.GL) && s.GL >= 0.0 &&
		finite(s.ENa) && finite(s.EK) && finite(s.EQ) && finite(s.EL) &&
		finite(s.TauQ) && s.TauQ > 0.0 && finite(s.Dt) && s.Dt > 0.0 &&
		finite(s.VThreshold)
}

func (s YamadaNeuronState) derivatives(v, n, q, iExt float64) (yamadaDerivative, bool) {
	if !finite(v) || !finite(n) || n < 0.0 || n > 1.0 || !finite(q) || q < 0.0 || q > 1.0 || !finite(iExt) {
		return yamadaDerivative{}, false
	}
	mInf := yamadaSigmoid((v + 30.0) / 9.5)
	nInf := yamadaSigmoid((v + 30.0) / 10.0)
	qInf := yamadaSigmoid((v + 50.0) / 10.0)
	tauN := yamadaTauN(v)
	iNa := s.GNa * math.Pow(mInf, 3.0) * (1.0 - n) * (v - s.ENa)
	iK := s.GK * math.Pow(n, 4.0) * (v - s.EK)
	iQ := s.GQ * q * (v - s.EQ)
	iL := s.GL * (v - s.EL)
	d := yamadaDerivative{
		v: -iNa - iK - iQ - iL + iExt,
		n: (nInf - n) / tauN,
		q: (qInf - q) / s.TauQ,
	}
	if !finite(mInf) || !finite(nInf) || !finite(qInf) || !finite(tauN) || !finite(iNa) || !finite(iK) || !finite(iQ) || !finite(iL) || !finite(d.v) || !finite(d.n) || !finite(d.q) {
		return yamadaDerivative{}, false
	}
	return d, true
}

// RK4Candidate returns the candidate state for one candidate-first RK4 timestep.
func (s YamadaNeuronState) RK4Candidate(iExt float64) (float64, float64, float64, bool) {
	if !finite(iExt) || !s.Valid() {
		return 0.0, 0.0, 0.0, false
	}

	k1, ok := s.derivatives(s.V, s.N, s.Q, iExt)
	if !ok {
		return 0.0, 0.0, 0.0, false
	}
	k2, ok := s.derivatives(s.V+0.5*s.Dt*k1.v, s.N+0.5*s.Dt*k1.n, s.Q+0.5*s.Dt*k1.q, iExt)
	if !ok {
		return 0.0, 0.0, 0.0, false
	}
	k3, ok := s.derivatives(s.V+0.5*s.Dt*k2.v, s.N+0.5*s.Dt*k2.n, s.Q+0.5*s.Dt*k2.q, iExt)
	if !ok {
		return 0.0, 0.0, 0.0, false
	}
	k4, ok := s.derivatives(s.V+s.Dt*k3.v, s.N+s.Dt*k3.n, s.Q+s.Dt*k3.q, iExt)
	if !ok {
		return 0.0, 0.0, 0.0, false
	}
	nextV := s.V + s.Dt*(k1.v+2.0*k2.v+2.0*k3.v+k4.v)/6.0
	nextN := s.N + s.Dt*(k1.n+2.0*k2.n+2.0*k3.n+k4.n)/6.0
	nextQ := s.Q + s.Dt*(k1.q+2.0*k2.q+2.0*k3.q+k4.q)/6.0
	if !finite(nextV) || !finite(nextN) || !finite(nextQ) || nextN < 0.0 || nextN > 1.0 || nextQ < 0.0 || nextQ > 1.0 {
		return 0.0, 0.0, 0.0, false
	}
	return nextV, nextN, nextQ, true
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *YamadaNeuronState) Step(iExt float64) int {
	if !finite(iExt) || !s.Valid() {
		return 0
	}

	vPrev := s.V
	nextV, nextN, nextQ, ok := s.RK4Candidate(iExt)
	if !ok {
		return 0
	}

	s.V = nextV
	s.N = nextN
	s.Q = nextQ
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// Reset restores dynamic state without changing parameters.
func (s *YamadaNeuronState) Reset() {
	s.V = -60.0
	s.N = 0.1
	s.Q = 0.0
}

// SimulateYamadaNeuron runs the neuron for n steps.
func SimulateYamadaNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewYamadaNeuron()
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
