// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - Go service tests for gamma motor neuron

package services

import (
	"math"
	"testing"
)

func TestGammaMotorContinuousRelaxation(t *testing.T) {
	n := NewGammaMotorNeuron()
	spike := n.Step(4.0)
	expectedV := -61.0 + (-65.0+61.0)*math.Exp(-0.5/8.0)
	expectedAdapt := 0.3 * (expectedV + 65.0) * (1.0 - math.Exp(-0.5/100.0))
	if spike != 0 {
		t.Fatalf("subthreshold drive produced spike")
	}
	if math.Abs(n.V-expectedV) > 1e-12 {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", n.V, expectedV)
	}
	if math.Abs(n.Adapt-expectedAdapt) > 1e-12 {
		t.Fatalf("adapt mismatch: got %.17g want %.17g", n.Adapt, expectedAdapt)
	}
}

func TestGammaMotorInvalidDrivePreservesState(t *testing.T) {
	n := NewGammaMotorNeuron()
	beforeV := n.V
	beforeAdapt := n.Adapt
	if spike := n.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid drive produced spike")
	}
	if n.V != beforeV || n.Adapt != beforeAdapt {
		t.Fatalf("invalid drive mutated state")
	}
}

func TestGammaMotorCorruptedStatePreservesState(t *testing.T) {
	n := NewGammaMotorNeuron()
	n.Tau = 0.0
	beforeV := n.V
	beforeAdapt := n.Adapt
	if spike := n.Step(20.0); spike != 0 {
		t.Fatalf("invalid state produced spike")
	}
	if n.V != beforeV || n.Adapt != beforeAdapt {
		t.Fatalf("invalid state mutated state")
	}
}
