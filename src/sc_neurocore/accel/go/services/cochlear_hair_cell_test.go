// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SPDX-FileCopyrightText: 2026 Arcane Sapience
//
// This file is part of SC-NeuroCore.
// Licensed under the GNU Affero General Public License v3.0 or later.
// See <https://www.gnu.org/licenses/>.

package services

import (
	"math"
	"testing"
)

func cochlearReferencePOpen(displacement, x0, delta float64) float64 {
	z := (displacement - x0) / delta
	if z >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-z))
	}
	ez := math.Exp(z)
	return ez / (1.0 + ez)
}

func cochlearReferenceVoltage(s *CochlearHairCellState, displacement float64) float64 {
	po := cochlearReferencePOpen(displacement, s.X0, s.Delta)
	gMET := s.GMax * po
	gTotal := s.GL + gMET
	vInf := (s.GL*s.EL + gMET*s.EMet) / gTotal
	return vInf + (s.V-vInf)*math.Exp(-(gTotal/s.Cap)*s.Dt)
}

func TestCochlearHairCellClosedFormRelaxation(t *testing.T) {
	state := NewCochlearHairCell()
	expected := cochlearReferenceVoltage(state, 0.0)
	spike := state.Step(0.0)
	if spike != 0 && spike != 1 {
		t.Fatalf("unexpected spike sentinel %d", spike)
	}
	if math.Abs(state.V-expected) > 1e-12 {
		t.Fatalf("closed-form voltage mismatch: got %.17g want %.17g", state.V, expected)
	}
}

func TestCochlearHairCellInvalidRuntimePreservesState(t *testing.T) {
	state := NewCochlearHairCell()
	state.V = -55.0
	state.GlutamateRelease = 0.125
	beforeV := state.V
	beforeRelease := state.GlutamateRelease
	state.Cap = -1.0
	if got := state.Step(0.25); got != -1 {
		t.Fatalf("expected invalid sentinel -1, got %d", got)
	}
	if state.V != beforeV || state.GlutamateRelease != beforeRelease {
		t.Fatalf("invalid runtime mutated state: V %.17g release %.17g", state.V, state.GlutamateRelease)
	}
}
