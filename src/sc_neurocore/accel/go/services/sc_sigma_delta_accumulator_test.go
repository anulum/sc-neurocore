// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

package services

import "testing"

func TestSCSigmaDeltaSignedRecurrence(t *testing.T) {
	s := NewSCSigmaDeltaAccumulator()
	e, err := s.Step(3.25)
	if err != nil || e != 1 || s.Sigma != 2.25 {
		t.Fatal(e, err, s.Sigma)
	}
	e, err = s.Step(-4.5)
	if err != nil || e != -1 || s.Sigma != -1.25 {
		t.Fatal(e, err, s.Sigma)
	}
}
