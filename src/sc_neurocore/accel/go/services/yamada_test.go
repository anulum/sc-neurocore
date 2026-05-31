// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Yamada RK4 service tests

package services

import (
	"math"
	"testing"
)

type yamadaState struct {
	v float64
	n float64
	q float64
}

func yamadaReferenceRK4(s YamadaNeuronState, current float64) yamadaState {
	rhs := func(v, n, q float64) yamadaState {
		mInf := yamadaSigmoid((v + 30.0) / 9.5)
		nInf := yamadaSigmoid((v + 30.0) / 10.0)
		qInf := yamadaSigmoid((v + 50.0) / 10.0)
		tauN := 1.0 + 7.5/(1.0+math.Exp((v+40.0)/12.0))
		iNa := s.GNa * math.Pow(mInf, 3.0) * (1.0 - n) * (v - s.ENa)
		iK := s.GK * math.Pow(n, 4.0) * (v - s.EK)
		iQ := s.GQ * q * (v - s.EQ)
		iL := s.GL * (v - s.EL)
		return yamadaState{
			v: -iNa - iK - iQ - iL + current,
			n: (nInf - n) / tauN,
			q: (qInf - q) / s.TauQ,
		}
	}

	k1 := rhs(s.V, s.N, s.Q)
	k2 := rhs(s.V+0.5*s.Dt*k1.v, s.N+0.5*s.Dt*k1.n, s.Q+0.5*s.Dt*k1.q)
	k3 := rhs(s.V+0.5*s.Dt*k2.v, s.N+0.5*s.Dt*k2.n, s.Q+0.5*s.Dt*k2.q)
	k4 := rhs(s.V+s.Dt*k3.v, s.N+s.Dt*k3.n, s.Q+s.Dt*k3.q)
	return yamadaState{
		v: s.V + s.Dt*(k1.v+2.0*k2.v+2.0*k3.v+k4.v)/6.0,
		n: s.N + s.Dt*(k1.n+2.0*k2.n+2.0*k3.n+k4.n)/6.0,
		q: s.Q + s.Dt*(k1.q+2.0*k2.q+2.0*k3.q+k4.q)/6.0,
	}
}

func TestYamadaRK4CandidateMatchesReference(t *testing.T) {
	s := NewYamadaNeuron()
	s.V = -52.0
	s.N = 0.22
	s.Q = 0.08
	s.Dt = 0.025
	expected := yamadaReferenceRK4(*s, 18.0)

	nextV, nextN, nextQ, ok := s.RK4Candidate(18.0)

	if !ok {
		t.Fatal("expected valid RK4 candidate")
	}
	if math.Abs(nextV-expected.v) > 1e-12 || math.Abs(nextN-expected.n) > 1e-14 || math.Abs(nextQ-expected.q) > 1e-14 {
		t.Fatalf("candidate mismatch: got (%.17g, %.17g, %.17g), want (%.17g, %.17g, %.17g)", nextV, nextN, nextQ, expected.v, expected.n, expected.q)
	}
}

func TestYamadaStepCommitsRK4Candidate(t *testing.T) {
	s := NewYamadaNeuron()
	s.V = -52.0
	s.N = 0.22
	s.Q = 0.08
	s.Dt = 0.025
	expected := yamadaReferenceRK4(*s, 18.0)

	s.Step(18.0)

	if math.Abs(s.V-expected.v) > 1e-12 || math.Abs(s.N-expected.n) > 1e-14 || math.Abs(s.Q-expected.q) > 1e-14 {
		t.Fatalf("state mismatch: got (%.17g, %.17g, %.17g), want (%.17g, %.17g, %.17g)", s.V, s.N, s.Q, expected.v, expected.n, expected.q)
	}
}

func TestYamadaInvalidCandidatePreservesState(t *testing.T) {
	s := NewYamadaNeuron()
	s.V = -55.0
	s.N = 0.2
	s.Q = 0.1
	s.Dt = 1.0e308

	if spike := s.Step(1.0e308); spike != 0 {
		t.Fatalf("invalid candidate reported spike %d", spike)
	}
	if s.V != -55.0 || s.N != 0.2 || s.Q != 0.1 {
		t.Fatalf("invalid candidate mutated state to (%.17g, %.17g, %.17g)", s.V, s.N, s.Q)
	}
}
