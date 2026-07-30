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

func TestMcKeanSourceAtomic(t *testing.T) {
	s := NewMcKeanNeuron()
	if s.Step(3) != 1 {
		t.Fatal("expected switching event")
	}
	v, w := s.V, s.W
	if s.Step(math.NaN()) != -1 || s.V != v || s.W != w {
		t.Fatal("invalid transition mutated state")
	}
}
