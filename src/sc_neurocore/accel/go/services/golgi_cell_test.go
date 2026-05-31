// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore GolgiCell service tests

package services

import (
	"math"
	"testing"
)

func TestGolgiCellInvalidCurrentPreservesState(t *testing.T) {
	cell := NewGolgiCell()
	for i := 0; i < 10; i++ {
		cell.Step(5.0)
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

func TestGolgiCellExcessCurrentPreservesState(t *testing.T) {
	cell := NewGolgiCell()
	before := *cell

	if spike := cell.Step(1.0e8); spike != 0 {
		t.Fatalf("excess current must not spike, got %d", spike)
	}
	if *cell != before {
		t.Fatalf("excess current mutated state: got %#v want %#v", *cell, before)
	}
}

func TestGolgiCellAllCurrentsBoundedAndCalciumActive(t *testing.T) {
	cell := NewGolgiCell()
	baselineCa := cell.Ca
	spikes := 0
	for i := 0; i < 2000; i++ {
		spikes += cell.Step(10.0)
	}

	if spikes <= 0 {
		t.Fatalf("Golgi cell should spike with sustained excitatory current")
	}
	if cell.Ca <= baselineCa {
		t.Fatalf("calcium should accumulate during spiking: got %.17g baseline %.17g", cell.Ca, baselineCa)
	}
	for name, gate := range map[string]float64{
		"M": cell.M, "H": cell.H, "PNa": cell.PNa, "N": cell.N, "A": cell.A, "B": cell.B,
		"W": cell.W, "MT": cell.MT, "S": cell.S, "CN": cell.CN, "R": cell.R,
	} {
		if gate < 0.0 || gate > 1.0 {
			t.Fatalf("%s gate out of bounds: %.17g", name, gate)
		}
	}
	if cell.V < -100.0 || cell.V > 60.0 {
		t.Fatalf("voltage outside guard band: %.17g", cell.V)
	}
}
