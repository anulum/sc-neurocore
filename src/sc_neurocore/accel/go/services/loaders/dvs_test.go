// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

package loaders_test

import (
	"github.com/anulum/sc-neurocore/accel/services/loaders"
	"testing"
)

// TestDVSInvalidPublicParameters checks actual API refusals unavailable in a NUL-free argv.
func TestDVSInvalidPublicParameters(t *testing.T) {
	for _, input := range []struct {
		path   string
		budget int
	}{
		{"", 0}, {"file\x00suffix", 0}, {"file", -1}, {t.TempDir(), 0},
	} {
		events, err := loaders.ReadDVSRecording(input.path, input.budget)
		if err == nil || events != nil {
			t.Fatalf("invalid input returned events: %v, %v", events, err)
		}
	}
}
