// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

// Package main exposes the native Go DVS recording reader as a binary CLI.
package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/anulum/sc-neurocore/accel/services/loaders"
)

// main writes DVS1, a uint64 value count and little-endian float64 values.
// Arguments are an NPY path, returned matrix byte budget and optional expected parent PID.
// This Linux command exits124 after 30 seconds, including blocked file/output I/O.
// Supplying the parent's PID closes the startup race; otherwise its current PID is used.
// Parent death kills the process; the caller retains responsibility for reaping it.
// Input refusal returns status1 and stderr diagnostics without stdout.
// Output failures may leave a partial frame and must be treated as unsuccessful.
func main() {
	if len(os.Args) != 3 && len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "expected DVS recording path, event budget and optional parent PID")
		os.Exit(1)
	}
	budget, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	parent := os.Getppid()
	if len(os.Args) == 4 {
		parent, err = strconv.Atoi(os.Args[3])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(124)
		}
	}
	if err = armParentDeath(parent); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(124)
	}
	deadline := time.AfterFunc(30*time.Second, func() { os.Exit(124) })
	defer deadline.Stop()
	events, err := loaders.ReadDVSRecording(os.Args[1], budget)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if _, err = os.Stdout.Write([]byte("DVS1")); err == nil {
		err = binary.Write(os.Stdout, binary.LittleEndian, uint64(len(events)))
	}
	if err == nil {
		err = binary.Write(os.Stdout, binary.LittleEndian, events)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
