// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

//go:build linux

package main

import (
	"errors"
	"os"
	"runtime"
	"syscall"
)

// armParentDeath binds the CLI goroutine to the thread carrying its Linux parent guard.
// It remains locked until process exit. The parent PID is checked before and after
// prctl to refuse startup after the expected parent died. Credential changes and
// uninterruptible kernel operations are outside this CLI's lifetime contract.
func armParentDeath(expectedParent int) error {
	runtime.LockOSThread()
	if expectedParent <= 0 || os.Getppid() != expectedParent {
		return errors.New("DVS expected parent is no longer present")
	}
	_, _, errno := syscall.Syscall6(syscall.SYS_PRCTL, 1, uintptr(syscall.SIGKILL), 0, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	if os.Getppid() != expectedParent {
		return errors.New("DVS parent changed while arming its guard")
	}
	return nil
}
