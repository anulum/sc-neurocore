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

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *YamadaNeuronState) Step(iExt float64) int {
	if !finite(iExt) || !s.Valid() {
		return 0
	}

	vPrev := s.V
	mInf := yamadaSigmoid((s.V + 30.0) / 9.5)
	nInf := yamadaSigmoid((s.V + 30.0) / 10.0)
	qInf := yamadaSigmoid((s.V + 50.0) / 10.0)
	tauN := 1.0 + 7.5/(1.0+math.Exp((s.V+40.0)/12.0))
	iNa := s.GNa * math.Pow(mInf, 3.0) * (1.0 - s.N) * (s.V - s.ENa)
	iK := s.GK * math.Pow(s.N, 4.0) * (s.V - s.EK)
	iQ := s.GQ * s.Q * (s.V - s.EQ)
	iL := s.GL * (s.V - s.EL)
	dv := (-iNa - iK - iQ - iL + iExt) * s.Dt
	dn := (nInf - s.N) / tauN * s.Dt
	dq := (qInf - s.Q) / s.TauQ * s.Dt
	nextV := s.V + dv
	nextN := s.N + dn
	nextQ := s.Q + dq
	if !finite(mInf) || !finite(nInf) || !finite(qInf) || !finite(tauN) || !finite(iNa) || !finite(iK) || !finite(iQ) || !finite(iL) || !finite(dv) || !finite(dn) || !finite(dq) || !finite(nextV) || !finite(nextN) || !finite(nextQ) || nextN < 0.0 || nextN > 1.0 || nextQ < 0.0 || nextQ > 1.0 {
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
