// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Opt-in MAT native coverage export

//go:build coverage

package main

import "C"

import "github.com/anulum/sc-neurocore/accel/internal/coverageprofile"

// write_coverage_snapshot returns zero only after both native writers succeed.
// A null path, unavailable directory or uninstrumented image returns one.
// This symbol is absent from ordinary production builds.
//
//export write_coverage_snapshot
func write_coverage_snapshot(path *C.char) C.int {
	if coverageprofile.Write(C.GoString(path)) != nil {
		return 1
	}
	return 0
}
