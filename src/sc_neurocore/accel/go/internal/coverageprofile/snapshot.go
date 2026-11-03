// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Opt-in native Go coverage snapshots

//go:build coverage

// Package coverageprofile writes native counters for diagnostic builds only.
package coverageprofile

import (
	"errors"
	"runtime/coverage"
)

// Write retains metadata and counters, reporting either native writer failure.
// The caller supplies an existing directory. Ordinary builds exclude this package.
func Write(directory string) error {
	return errors.Join(coverage.WriteMetaDir(directory), coverage.WriteCountersDir(directory))
}
