// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go GranuleCell behavioural tests

package services

import (
	"math"
	"testing"
)

func TestGranuleCellStepPreservesBounds(t *testing.T) {
	cell := NewGranuleCell()

	for i := 0; i < 200; i++ {
		spike := cell.Step(0.0)
		if spike != 0 && spike != 1 {
			t.Fatalf("spike indicator must be binary, got %d", spike)
		}
	}

	if math.IsNaN(cell.V) || math.IsInf(cell.V, 0) || cell.V < -100.0 || cell.V > 60.0 {
		t.Fatalf("membrane potential must stay finite and bounded, got %f", cell.V)
	}
	for name, gate := range map[string]float64{
		"M":  cell.M,
		"H":  cell.H,
		"N":  cell.N,
		"A":  cell.A,
		"B":  cell.B,
		"MT": cell.MT,
		"S":  cell.S,
		"R":  cell.R,
	} {
		if gate < 0.0 || gate > 1.0 {
			t.Fatalf("%s gate must stay in [0, 1], got %f", name, gate)
		}
	}
	if cell.Ca < 0.0 {
		t.Fatalf("calcium concentration must be non-negative, got %f", cell.Ca)
	}
}

func TestGranuleCellTonicGabaSuppressesDrive(t *testing.T) {
	withGaba := NewGranuleCell()
	withoutGaba := NewGranuleCell()
	withoutGaba.GTonic = 0.0
	spikesWith := 0
	spikesWithout := 0

	for i := 0; i < 10000; i++ {
		spikesWith += withGaba.Step(8.0)
		spikesWithout += withoutGaba.Step(8.0)
	}

	if spikesWithout <= spikesWith {
		t.Fatalf("removing tonic GABA should increase firing: without=%d with=%d", spikesWithout, spikesWith)
	}
}

func TestGranuleCellGateAndCalciumKineticsUseClosedFormRelaxation(t *testing.T) {
	cell := NewGranuleCell()
	cell.GNa = 0.0
	cell.GKdr = 0.0
	cell.GKa = 0.0
	cell.GT = 0.0
	cell.GKca = 0.0
	cell.GH = 0.0
	cell.GL = 0.0
	cell.GTonic = 0.0
	cell.Gain = 0.0
	cell.SubSteps = 1
	v0, m0, h0, n0, a0 := cell.V, cell.M, cell.H, cell.N, cell.A
	b0, mt0, s0, ca0, r0 := cell.B, cell.MT, cell.S, cell.Ca, cell.R
	mInf := boltzGranule(v0, -30.0, 7.0)
	tauM := 0.1 + 0.3/math.Max(0.01, 1.0+math.Pow((v0+30.0)/10.0, 2.0))
	hInf := boltzGranule(v0, -52.0, -6.0)
	tauH := 0.5 + 5.0/math.Max(0.01, 1.0+math.Pow((v0+50.0)/15.0, 2.0))
	nInf := boltzGranule(v0, -35.0, 8.0)
	tauN := 1.0 + 5.0/math.Max(0.01, 1.0+math.Pow((v0+35.0)/15.0, 2.0))
	aInf := boltzGranule(v0, -50.0, 20.0)
	bInf := boltzGranule(v0, -70.0, -6.0)
	mtInf := boltzGranule(v0, -52.0, 5.0)
	sInf := boltzGranule(v0, -60.0, -6.5)
	tauS := 20.0 + 50.0/math.Max(0.01, 1.0+math.Pow((v0+65.0)/10.0, 2.0))
	rInf := boltzGranule(v0, -80.0, -10.0)
	tauR := 50.0 + 200.0/math.Max(0.01, 1.0+math.Pow((v0+80.0)/20.0, 2.0))

	cell.Step(0.0)

	expectCloseGranule(t, cell.V, v0, "V")
	expectCloseGranule(t, cell.M, exactRelaxGranule(m0, mInf, tauM, cell.Dt), "M")
	expectCloseGranule(t, cell.H, exactRelaxGranule(h0, hInf, tauH, cell.Dt), "H")
	expectCloseGranule(t, cell.N, exactRelaxGranule(n0, nInf, tauN, cell.Dt), "N")
	expectCloseGranule(t, cell.A, exactRelaxGranule(a0, aInf, 2.0, cell.Dt), "A")
	expectCloseGranule(t, cell.B, exactRelaxGranule(b0, bInf, 50.0, cell.Dt), "B")
	expectCloseGranule(t, cell.MT, exactRelaxGranule(mt0, mtInf, 1.0, cell.Dt), "MT")
	expectCloseGranule(t, cell.S, exactRelaxGranule(s0, sInf, tauS, cell.Dt), "S")
	expectCloseGranule(t, cell.Ca, exactRelaxGranule(ca0, 0.0, cell.TauCa, cell.Dt), "Ca")
	expectCloseGranule(t, cell.R, exactRelaxGranule(r0, rInf, tauR, cell.Dt), "R")
}

func expectCloseGranule(t *testing.T, observed, expected float64, name string) {
	t.Helper()
	if math.Abs(observed-expected) > 1e-12 {
		t.Fatalf("%s mismatch: observed %.17g expected %.17g", name, observed, expected)
	}
}

func TestGranuleCellInvalidDrivePreservesState(t *testing.T) {
	cell := NewGranuleCell()
	beforeV := cell.V
	beforeCa := cell.Ca
	beforeS := cell.S

	if spike := cell.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid drive must not spike, got %d", spike)
	}
	if cell.V != beforeV || cell.Ca != beforeCa || cell.S != beforeS {
		t.Fatalf("invalid drive mutated state")
	}
}

func TestGranuleCellCorruptedGatePreservesState(t *testing.T) {
	cell := NewGranuleCell()
	cell.M = -0.1
	beforeV := cell.V
	beforeM := cell.M
	beforeCa := cell.Ca

	if spike := cell.Step(8.0); spike != 0 {
		t.Fatalf("corrupted state must not spike, got %d", spike)
	}
	if cell.V != beforeV || cell.M != beforeM || cell.Ca != beforeCa {
		t.Fatalf("corrupted state mutated during fail-closed step")
	}
}
