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
	if n.Ca < 0.0 || !math.IsInf(n.V, 0) && !math.IsNaN(n.V) && (n.V < -100.0 || n.V > 60.0) {
		t.Fatalf("invalid state after step: V=%.17g Ca=%.17g", n.V, n.Ca)
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
