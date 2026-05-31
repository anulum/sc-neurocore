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

func TestFitzHughRinzelCurrentBalance(t *testing.T) {
	s := NewFitzHughRinzelNeuron()
	s.V = -1.0
	s.W = 0.2
	s.Y = 0.1
	spike := s.Step(0.5)
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(s.V-(-1.0266666666666666)) > 1e-12 {
		t.Fatalf("unexpected v: %.17g", s.V)
	}
	if math.Abs(s.W-0.19632) > 1e-12 {
		t.Fatalf("unexpected w: %.17g", s.W)
	}
	if math.Abs(s.Y-0.10000125) > 1e-12 {
		t.Fatalf("unexpected y: %.17g", s.Y)
	}
}

func TestFitzHughRinzelInvalidCurrentPreservesState(t *testing.T) {
	s := NewFitzHughRinzelNeuron()
	before := [3]float64{s.V, s.W, s.Y}
	if spike := s.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid current produced spike: %d", spike)
	}
	after := [3]float64{s.V, s.W, s.Y}
	if after != before {
		t.Fatalf("state mutated: before=%v after=%v", before, after)
	}
}

func TestFitzHughRinzelOverflowCandidatePreservesState(t *testing.T) {
	s := NewFitzHughRinzelNeuron()
	s.V = 1.0e155
	before := [3]float64{s.V, s.W, s.Y}
	if spike := s.Step(0.5); spike != 0 {
		t.Fatalf("overflow candidate produced spike: %d", spike)
	}
	after := [3]float64{s.V, s.W, s.Y}
	if after != before {
		t.Fatalf("state mutated: before=%v after=%v", before, after)
	}
}
