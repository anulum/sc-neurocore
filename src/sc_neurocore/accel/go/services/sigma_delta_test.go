// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

package services

import (
	"math"
	"testing"
)

func TestSigmaDeltaSourceAndAtomicFailure(t *testing.T) {
	s := NewSigmaDeltaNeuron()
	s.Sigma = 0.49
	e, err := s.Step(0.2)
	if err != nil || e != 1 {
		t.Fatalf("event=%d err=%v", e, err)
	}
	beforeS, beforeR := s.Sigma, s.Reconstruction
	if _, err = s.Step(math.NaN()); err == nil || s.Sigma != beforeS || s.Reconstruction != beforeR {
		t.Fatal("invalid transition was not atomic")
	}
}
