// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go StellateCell behavioural tests

package services

import (
	"math"
	"testing"
)

func TestStellateCellStepPreservesBounds(t *testing.T) {
	cell := NewStellateCell()

	for i := 0; i < 200; i++ {
		spike := cell.Step(0.0)
		if spike != 0 && spike != 1 {
			t.Fatalf("spike indicator must be binary, got %d", spike)
		}
	}

	if math.IsNaN(cell.V) || math.IsInf(cell.V, 0) || cell.V < -100.0 || cell.V > 60.0 {
		t.Fatalf("membrane potential must stay finite and bounded, got %f", cell.V)
	}
	for name, gate := range map[string]float64{"H": cell.H, "N": cell.N, "P": cell.P} {
		if gate < 0.0 || gate > 1.0 {
			t.Fatalf("%s gate must stay in [0, 1], got %f", name, gate)
		}
	}
}

func TestStellateCellKv3GateActivatesWithDepolarisation(t *testing.T) {
	resting := NewStellateCell()
	depolarised := NewStellateCell()

	for i := 0; i < 100; i++ {
		resting.Step(0.0)
		depolarised.Step(8.0)
	}

	if depolarised.P <= resting.P {
		t.Fatalf("Kv3 gate should activate more during depolarisation: depolarised=%f resting=%f", depolarised.P, resting.P)
	}
}

func TestStellateCellGateKineticsUseClosedFormRelaxation(t *testing.T) {
	cell := NewStellateCell()
	cell.GNa = 0.0
	cell.GK = 0.0
	cell.GKv3 = 0.0
	cell.GL = 0.0
	cell.Gain = 0.0
	cell.SubSteps = 1
	v0, h0, n0, p0 := cell.V, cell.H, cell.N, cell.P
	alphaH := 0.07 * safeExpStellate(-(v0+58.0)/20.0)
	betaH := boltzStellate(v0, -28.0, 10.0)
	alphaN := safeRateStellate(0.01, 34.0, v0, 10.0, 0.1)
	betaN := 0.125 * safeExpStellate(-(v0+44.0)/80.0)
	pInf := boltzStellate(v0, -10.0, 10.0)
	tauP := 1.0 + 4.0/(1.0+safeExpStellate((v0+20.0)/15.0))

	cell.Step(0.0)

	expectCloseStellate(t, cell.V, v0, "V")
	expectCloseStellate(t, cell.H, exactHHGateStellate(h0, alphaH, betaH, cell.Phi, cell.Dt), "H")
	expectCloseStellate(t, cell.N, exactHHGateStellate(n0, alphaN, betaN, cell.Phi, cell.Dt), "N")
	expectCloseStellate(t, cell.P, exactRelaxStellate(p0, pInf, tauP, cell.Dt), "P")
}

func expectCloseStellate(t *testing.T, observed, expected float64, name string) {
	t.Helper()
	if math.Abs(observed-expected) > 1e-12 {
		t.Fatalf("%s mismatch: observed %.17g expected %.17g", name, observed, expected)
	}
}

func TestStellateCellInvalidDrivePreservesState(t *testing.T) {
	cell := NewStellateCell()
	beforeV := cell.V
	beforeH := cell.H
	beforeN := cell.N
	beforeP := cell.P

	if spike := cell.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid drive must not spike, got %d", spike)
	}
	if cell.V != beforeV || cell.H != beforeH || cell.N != beforeN || cell.P != beforeP {
		t.Fatalf("invalid drive mutated state")
	}
}

func TestStellateCellCorruptedGatePreservesState(t *testing.T) {
	cell := NewStellateCell()
	cell.H = -0.1
	beforeV := cell.V
	beforeH := cell.H
	beforeN := cell.N
	beforeP := cell.P

	if spike := cell.Step(8.0); spike != 0 {
		t.Fatalf("corrupted state must not spike, got %d", spike)
	}
	if cell.V != beforeV || cell.H != beforeH || cell.N != beforeN || cell.P != beforeP {
		t.Fatalf("corrupted state mutated during fail-closed step")
	}
}
