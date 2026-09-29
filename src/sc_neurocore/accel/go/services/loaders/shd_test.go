// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Indexed SHD HDF5 recordings

//go:build hdf5

package loaders

import "testing"

func TestReadSHDRecordingRejectsInvalidParameters(t *testing.T) {
	for _, input := range []struct {
		path          string
		index, budget int
	}{
		{"", 0, 1024}, {"file\x00suffix", 0, 1024},
		{"file", -1, 1024}, {"file", 0, -1},
	} {
		recording, err := ReadSHDRecording(input.path, input.index, input.budget)
		if err == nil || recording.Events != nil || recording.Label != 0 {
			t.Fatalf("invalid parameters returned a recording: %#v, %v", recording, err)
		}
	}
}
