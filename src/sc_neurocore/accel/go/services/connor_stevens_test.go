// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

package services

import (
	"math"
	"testing"
)

type connorState struct {
	v float64
	m float64
	h float64
	n float64
	a float64
	b float64
}

func connorReferenceRate(scale, shift, v, denom float64) float64 {
	delta := v + shift
	x := delta / denom
	if math.Abs(x) < 1e-9 {
		return scale * denom
	}
	return scale * delta / (1.0 - math.Exp(-x))
}

func connorReferenceDerivatives(s connorState, c ConnorStevensNeuron, current float64) connorState {
	alphaM := connorReferenceRate(0.38, 29.7, s.v, 10.0)
	betaM := 15.2 * math.Exp(-(s.v+54.7)/18.0)
	alphaH := 0.266 * math.Exp(-(s.v+48.0)/20.0)
	betaH := 3.8 / (1.0 + math.Exp(-(s.v+18.0)/10.0))
	alphaN := connorReferenceRate(0.02, 45.7, s.v, 10.0)
	betaN := 0.25 * math.Exp(-(s.v+55.7)/80.0)
	aInf := math.Pow(0.0761*math.Exp((s.v+94.22)/31.84)/(1.0+math.Exp((s.v+1.17)/28.93)), 1.0/3.0)
	tauA := 0.3632 + 1.158/(1.0+math.Exp((s.v+55.96)/20.12))
	bInf := math.Pow(1.0/(1.0+math.Exp((s.v+53.3)/14.54)), 4)
	tauB := 1.24 + 2.678/(1.0+math.Exp((s.v+50.0)/16.027))
	iNa := c.GNa * math.Pow(s.m, 3) * s.h * (s.v - c.ENa)
	iK := c.GK * math.Pow(s.n, 4) * (s.v - c.EK)
	iA := c.GA * math.Pow(s.a, 3) * s.b * (s.v - c.EA)
	iL := c.GL * (s.v - c.EL)
	return connorState{
		v: (-iNa - iK - iA - iL + current) / c.CM,
		m: alphaM*(1.0-s.m) - betaM*s.m,
		h: alphaH*(1.0-s.h) - betaH*s.h,
		n: alphaN*(1.0-s.n) - betaN*s.n,
		a: (aInf - s.a) / tauA,
		b: (bInf - s.b) / tauB,
	}
}

func connorReferenceRK4(c ConnorStevensNeuron, current float64) connorState {
	state := connorState{v: c.V, m: c.M, h: c.H, n: c.N, a: c.A, b: c.B}
	substeps := int(1.0 / math.Max(c.Dt, 0.001))
	for i := 0; i < substeps; i++ {
		k1 := connorReferenceDerivatives(state, c, current)
		k2 := connorReferenceDerivatives(connorState{v: state.v + 0.5*c.Dt*k1.v, m: state.m + 0.5*c.Dt*k1.m, h: state.h + 0.5*c.Dt*k1.h, n: state.n + 0.5*c.Dt*k1.n, a: state.a + 0.5*c.Dt*k1.a, b: state.b + 0.5*c.Dt*k1.b}, c, current)
		k3 := connorReferenceDerivatives(connorState{v: state.v + 0.5*c.Dt*k2.v, m: state.m + 0.5*c.Dt*k2.m, h: state.h + 0.5*c.Dt*k2.h, n: state.n + 0.5*c.Dt*k2.n, a: state.a + 0.5*c.Dt*k2.a, b: state.b + 0.5*c.Dt*k2.b}, c, current)
		k4 := connorReferenceDerivatives(connorState{v: state.v + c.Dt*k3.v, m: state.m + c.Dt*k3.m, h: state.h + c.Dt*k3.h, n: state.n + c.Dt*k3.n, a: state.a + c.Dt*k3.a, b: state.b + c.Dt*k3.b}, c, current)
		state = connorState{
			v: state.v + c.Dt*(k1.v+2.0*k2.v+2.0*k3.v+k4.v)/6.0,
			m: state.m + c.Dt*(k1.m+2.0*k2.m+2.0*k3.m+k4.m)/6.0,
			h: state.h + c.Dt*(k1.h+2.0*k2.h+2.0*k3.h+k4.h)/6.0,
			n: state.n + c.Dt*(k1.n+2.0*k2.n+2.0*k3.n+k4.n)/6.0,
			a: state.a + c.Dt*(k1.a+2.0*k2.a+2.0*k3.a+k4.a)/6.0,
			b: state.b + c.Dt*(k1.b+2.0*k2.b+2.0*k3.b+k4.b)/6.0,
		}
	}
	return state
}

func TestConnorStevensMatchesIndependentRK4(t *testing.T) {
	neuron := NewConnorStevensNeuron()
	neuron.V = -62.0
	neuron.M = 0.05
	neuron.H = 0.84
	neuron.N = 0.22
	neuron.A = 0.41
	neuron.B = 0.27
	neuron.Dt = 0.02
	expected := connorReferenceRK4(*neuron, 8.5)

	spike, err := neuron.Step(8.5)
	if err != nil {
		t.Fatalf("step returned error: %v", err)
	}
	if spike != 0 && spike != 1 {
		t.Fatalf("invalid spike value %d", spike)
	}
	if math.Abs(neuron.V-expected.v) > 1e-10 || math.Abs(neuron.M-expected.m) > 1e-10 || math.Abs(neuron.H-expected.h) > 1e-10 || math.Abs(neuron.N-expected.n) > 1e-10 || math.Abs(neuron.A-expected.a) > 1e-10 || math.Abs(neuron.B-expected.b) > 1e-10 {
		t.Fatalf("state mismatch got (%g,%g,%g,%g,%g,%g) expected (%g,%g,%g,%g,%g,%g)", neuron.V, neuron.M, neuron.H, neuron.N, neuron.A, neuron.B, expected.v, expected.m, expected.h, expected.n, expected.a, expected.b)
	}
}

func TestConnorStevensInvalidCurrentPreservesState(t *testing.T) {
	neuron := NewConnorStevensNeuron()
	before := *neuron
	if _, err := neuron.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid current error")
	}
	if *neuron != before {
		t.Fatalf("state mutated after invalid current")
	}
}

func TestConnorStevensCorruptStatePreservesState(t *testing.T) {
	neuron := NewConnorStevensNeuron()
	neuron.B = math.Inf(1)
	before := *neuron
	if _, err := neuron.Step(6.0); err == nil {
		t.Fatal("expected invalid state error")
	}
	if *neuron != before {
		t.Fatalf("state mutated after invalid state")
	}
}
