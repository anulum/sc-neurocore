// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - Go service tests for DCN neuron

package services

import (
	"math"
	"testing"
)

func TestDCNStepPreservesPhysicalBounds(t *testing.T) {
	n := NewDCNNeuron()
	spike := n.Step(0.0)
	if spike != 0 && spike != 1 {
		t.Fatalf("invalid spike value %d", spike)
	}
	for name, value := range map[string]float64{
		"H": n.H, "N": n.N, "P": n.P, "S": n.S, "R": n.R,
	} {
		if value < 0.0 || value > 1.0 {
			t.Fatalf("%s gate out of bounds: %.17g", name, value)
		}
	}
	if n.Ca < 0.0 || math.IsInf(n.V, 0) || math.IsNaN(n.V) || n.V < -100.0 || n.V > 60.0 {
		t.Fatalf("invalid state after step: V=%.17g Ca=%.17g", n.V, n.Ca)
	}
}

func TestDCNGateAndCalciumKineticsUseClosedFormRelaxation(t *testing.T) {
	n := NewDCNNeuron()
	n.GNa = 0.0
	n.GNap = 0.0
	n.GK = 0.0
	n.GT = 0.0
	n.GAhp = 0.0
	n.GH = 0.0
	n.GL = 0.0
	n.Gain = 0.0
	v0, h0, gateN0, p0, s0, r0, ca0 := n.V, n.H, n.N, n.P, n.S, n.R, n.Ca

	alphaH := 0.07 * math.Exp(-(v0+58.0)/20.0)
	betaH := 1.0 / (1.0 + math.Exp(-(v0+28.0)/10.0))
	alphaN := safeRateDCN(0.01, 34.0, v0, 10.0, 0.1)
	betaN := 0.125 * math.Exp(-(v0+44.0)/80.0)
	pInf := 1.0 / (1.0 + math.Exp(-(v0+48.0)/5.0))
	tauP := 5.0 + 15.0/math.Max(0.01, 1.0+math.Pow((v0+48.0)/10.0, 2))
	sInf := 1.0 / (1.0 + math.Exp((v0+60.0)/6.5))
	tauS := 20.0 + 50.0/(1.0+math.Exp((v0+65.0)/10.0))
	rInf := 1.0 / (1.0 + math.Exp((v0+80.0)/10.0))
	tauR := 100.0 + 200.0/(1.0+math.Exp((v0+70.0)/10.0))

	n.Step(0.0)

	expectCloseDCN(t, n.V, v0, "V")
	expectCloseDCN(t, n.H, exactHHGateDCN(h0, alphaH, betaH, n.Phi, n.Dt), "H")
	expectCloseDCN(t, n.N, exactHHGateDCN(gateN0, alphaN, betaN, n.Phi, n.Dt), "N")
	expectCloseDCN(t, n.P, exactRelaxDCN(p0, pInf, tauP, n.Dt), "P")
	expectCloseDCN(t, n.S, exactRelaxDCN(s0, sInf, tauS, n.Dt), "S")
	expectCloseDCN(t, n.R, exactRelaxDCN(r0, rInf, tauR, n.Dt), "R")
	expectCloseDCN(t, n.Ca, exactRelaxDCN(ca0, 0.0, n.TauCa, n.Dt), "Ca")
}

func expectCloseDCN(t *testing.T, observed, expected float64, name string) {
	t.Helper()
	if math.Abs(observed-expected) > 1e-12 {
		t.Fatalf("%s mismatch: observed %.17g expected %.17g", name, observed, expected)
	}
}

func TestDCNInvalidDrivePreservesState(t *testing.T) {
	n := NewDCNNeuron()
	beforeV := n.V
	beforeCa := n.Ca
	if spike := n.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid drive produced spike")
	}
	if n.V != beforeV || n.Ca != beforeCa {
		t.Fatalf("invalid drive mutated state")
	}
}

func TestDCNCorruptedStatePreservesState(t *testing.T) {
	n := NewDCNNeuron()
	n.H = -0.1
	beforeV := n.V
	beforeCa := n.Ca
	if spike := n.Step(5.0); spike != 0 {
		t.Fatalf("invalid state produced spike")
	}
	if n.V != beforeV || n.Ca != beforeCa {
		t.Fatalf("invalid state mutated state")
	}
}
