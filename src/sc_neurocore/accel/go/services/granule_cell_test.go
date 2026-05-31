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
