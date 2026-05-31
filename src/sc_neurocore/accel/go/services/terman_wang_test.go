// SPDX-License-Identifier: AGPL-3.0-or-later
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

func TestTermanWangCurrentBalance(t *testing.T) {
	s := NewTermanWangOscillator()
	v0, w0 := s.V, s.W
	f := 3.0*v0 - v0*v0*v0 + 2.0
	g := s.Alpha * (1.0 + math.Tanh(v0/s.Beta))
	expectedV := v0 + (f-w0+1.0+s.Rho)*s.Dt
	expectedW := w0 + s.Epsilon*(g-w0)*s.Dt

	spike, err := s.Step(1.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(s.V-expectedV) > 1e-12 {
		t.Fatalf("unexpected v: %.17g want %.17g", s.V, expectedV)
	}
	if math.Abs(s.W-expectedW) > 1e-12 {
		t.Fatalf("unexpected w: %.17g want %.17g", s.W, expectedW)
	}
}

func TestTermanWangInvalidCurrentPreservesState(t *testing.T) {
	s := NewTermanWangOscillator()
	before := [2]float64{s.V, s.W}
	if _, err := s.Step(math.NaN()); err == nil {
		t.Fatal("invalid current was accepted")
	}
	after := [2]float64{s.V, s.W}
	if after != before {
		t.Fatalf("state mutated: before=%v after=%v", before, after)
	}
}

func TestTermanWangOverflowCandidatePreservesState(t *testing.T) {
	s := NewTermanWangOscillator()
	s.V = 1.0e308
	before := [2]float64{s.V, s.W}
	if _, err := s.Step(1.0); err == nil {
		t.Fatal("overflow candidate was accepted")
	}
	after := [2]float64{s.V, s.W}
	if after != before {
		t.Fatalf("state mutated: before=%v after=%v", before, after)
	}
}
