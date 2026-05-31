// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore Renshaw cell service tests

package services

import (
	"math"
	"testing"
)

func renshawTestSafeRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

func renshawTestExactGate(previous, alpha, beta, phi, dt float64) float64 {
	total := phi * (alpha + beta)
	steady := alpha / (alpha + beta)
	return math.Max(0.0, math.Min(1.0, steady+(previous-steady)*math.Exp(-total*dt)))
}

func renshawTestExactAdapt(previous, steady, tau, dt float64) float64 {
	return math.Max(0.0, math.Min(1.0, steady+(previous-steady)*math.Exp(-dt/tau)))
}

func renshawReferenceStep(s RenshawCellState, iExt float64) RenshawCellState {
	nSub := int(0.5 / math.Max(s.Dt, 0.001))
	if nSub < 1 {
		nSub = 1
	}
	v := s.V
	h := s.H
	n := s.N
	adapt := s.Adapt
	for i := 0; i < nSub; i++ {
		am := renshawTestSafeRate(0.1, 35.0, v, 10.0, 1.0)
		bm := 4.0 * math.Exp(-(v+60.0)/18.0)
		ah := 0.07 * math.Exp(-(v+58.0)/20.0)
		bh := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
		an := renshawTestSafeRate(0.01, 34.0, v, 10.0, 0.1)
		bn := 0.125 * math.Exp(-(v+44.0)/80.0)
		mInf := am / (am + bm)
		h = renshawTestExactGate(h, ah, bh, s.Phi, s.Dt)
		n = renshawTestExactGate(n, an, bn, s.Phi, s.Dt)
		adaptInf := 1.0 / (1.0 + math.Exp(-(v+30.0)/5.0))
		adapt = renshawTestExactAdapt(adapt, adaptInf, s.TauAdapt, s.Dt)
		gNa := s.GNa * math.Pow(mInf, 3.0) * h
		gK := s.GK * math.Pow(n, 4.0)
		gAdapt := s.GAdapt * adapt
		gTotal := gNa + gK + gAdapt + s.GL
		steadyV := (iExt + gNa*s.ENa + gK*s.EK + gAdapt*s.EK + s.GL*s.EL) / gTotal
		v = steadyV + (v-steadyV)*math.Exp(-(gTotal/s.CM)*s.Dt)
	}
	s.V = v
	s.H = h
	s.N = n
	s.Adapt = adapt
	return s
}

func requireRenshawClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("%s mismatch: got %.17g want %.17g", name, got, want)
	}
}

func TestRenshawCellExactGateAndConductanceMembraneStep(t *testing.T) {
	cell := NewRenshawCell()
	expected := renshawReferenceStep(*NewRenshawCell(), 4.0)

	if spike := cell.Step(4.0); spike != 0 {
		t.Fatalf("first exact step should not cross threshold, got spike %d", spike)
	}

	requireRenshawClose(t, "V", cell.V, expected.V)
	requireRenshawClose(t, "H", cell.H, expected.H)
	requireRenshawClose(t, "N", cell.N, expected.N)
	requireRenshawClose(t, "Adapt", cell.Adapt, expected.Adapt)
}

func TestRenshawCellRejectsInvalidCurrentWithoutMutation(t *testing.T) {
	cell := NewRenshawCell()
	for i := 0; i < 20; i++ {
		cell.Step(4.0)
	}
	before := *cell

	if spike := cell.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid current must not spike, got %d", spike)
	}
	if *cell != before {
		t.Fatalf("invalid current mutated state: got %#v want %#v", *cell, before)
	}
	if spike := cell.Step(math.Inf(1)); spike != 0 {
		t.Fatalf("infinite current must not spike, got %d", spike)
	}
	if *cell != before {
		t.Fatalf("infinite current mutated state: got %#v want %#v", *cell, before)
	}
}

func TestRenshawCellRejectsExcessCurrentWithoutCorruption(t *testing.T) {
	cell := NewRenshawCell()
	before := *cell

	if spike := cell.Step(1.0e8); spike != 0 {
		t.Fatalf("excess current must not spike, got %d", spike)
	}
	if *cell != before {
		t.Fatalf("excess current mutated state: got %#v want %#v", *cell, before)
	}
}

func TestRenshawCellAdaptationRemainsBoundedAndIncreases(t *testing.T) {
	cell := NewRenshawCell()
	baseline := cell.Adapt
	spikes := 0
	for i := 0; i < 3000; i++ {
		spikes += cell.Step(4.0)
	}

	if spikes <= 0 {
		t.Fatalf("sustained collateral drive should elicit spikes")
	}
	if cell.Adapt <= baseline+0.01 {
		t.Fatalf("adaptation did not increase: got %.17g baseline %.17g", cell.Adapt, baseline)
	}
	if cell.H < 0.0 || cell.H > 1.0 || cell.N < 0.0 || cell.N > 1.0 || cell.Adapt < 0.0 || cell.Adapt > 1.0 {
		t.Fatalf("gates must remain bounded: H=%.17g N=%.17g Adapt=%.17g", cell.H, cell.N, cell.Adapt)
	}
	if cell.V < -150.0 || cell.V > 100.0 {
		t.Fatalf("voltage outside physiological guard band: %.17g", cell.V)
	}
}
