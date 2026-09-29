// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

//go:build !linux

package main

import "errors"

// armParentDeath refuses a guarded command where the Linux lifetime contract is unavailable.
// The public Go recording API remains usable independently of this guarded CLI.
func armParentDeath(expectedParent int) error {
	return errors.New("guarded DVS recording command requires Linux")
}
