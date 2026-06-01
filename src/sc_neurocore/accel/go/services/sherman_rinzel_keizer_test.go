// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Sherman-Rinzel-Keizer tests

package services

import (
	"math"
	"testing"
)

func TestShermanRinzelKeizerStepMatchesRK4Reference(t *testing.T) {
	s := NewShermanRinzelKeizerNeuron()
	spike := s.Step(5.0)
	if spike != 0 {
		t.Fatalf("expected subthreshold step, got %d", spike)
	}
	if math.Abs(s.V-(-54.24952703064663)) > 1e-12 {
		t.Fatalf("unexpected V %.15f", s.V)
	}
	if math.Abs(s.N-0.09468731121669713) > 1e-12 {
		t.Fatalf("unexpected N %.15f", s.N)
	}
	if math.Abs(s.S-0.10000523900468992) > 1e-12 {
		t.Fatalf("unexpected S %.15f", s.S)
	}
}

func TestShermanRinzelKeizerInvalidRuntimePreservesState(t *testing.T) {
	s := NewShermanRinzelKeizerNeuron()
	before := *s
	s.N = 1.2
	if spike := s.Step(5.0); spike != 0 {
		t.Fatalf("invalid state must not spike, got %d", spike)
	}
	if s.V != before.V || s.S != before.S || s.GCa != before.GCa {
		t.Fatalf("invalid state mutated unrelated fields: %+v", s)
	}
	if s.N != 1.2 {
		t.Fatalf("invalid gate should be preserved for caller diagnosis")
	}
}
