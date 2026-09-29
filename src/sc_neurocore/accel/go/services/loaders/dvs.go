// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

package loaders

import (
	"encoding/binary"
	"errors"
	"io"
	"os"
	"strings"
	"unicode/utf8"
)

// ReadDVSRecording reads one real numeric NPY (N,4) matrix into owned row-major doubles.
// Versions 1.0/2.0/3.0, either byte order and C/Fortran storage are supported.
// maximumBytes bounds returned event bytes; input buffers are additional memory.
// Header parsing is bounded to 10,000 bytes. Truncation and trailing content refuse.
// No timestamp rescaling, geometry validation, pickle or synthetic fallback occurs.
// Extended floats follow the host x86 80-bit representation, padded to 16 bytes.
func ReadDVSRecording(path string, maximumBytes int) (events []float64, err error) {
	if maximumBytes < 0 || path == "" || strings.ContainsRune(path, '\x00') {
		return nil, errors.New("invalid DVS path or event budget")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := file.Close(); err == nil && closeErr != nil {
			events = nil
			err = closeErr
		}
	}()
	prefix := make([]byte, 8)
	if _, err = io.ReadFull(file, prefix); err != nil {
		return nil, err
	}
	if string(prefix[:6]) != "\x93NUMPY" || prefix[7] != 0 || prefix[6] < 1 || prefix[6] > 3 {
		return nil, errors.New("invalid DVS NPY preamble")
	}
	width := 4
	if prefix[6] == 1 {
		width = 2
	}
	lengthRaw := make([]byte, width)
	if _, err = io.ReadFull(file, lengthRaw); err != nil {
		return nil, err
	}
	length := uint32(binary.LittleEndian.Uint16(lengthRaw))
	if width == 4 {
		length = binary.LittleEndian.Uint32(lengthRaw)
	}
	if length == 0 || length > 10000 {
		return nil, errors.New("invalid DVS NPY header length")
	}
	raw := make([]byte, int(length))
	if _, err = io.ReadFull(file, raw); err != nil {
		return nil, err
	}
	if raw[len(raw)-1] != '\n' || (prefix[6] == 3 && !utf8.Valid(raw)) {
		return nil, errors.New("invalid DVS NPY header encoding")
	}
	metadata, err := parseDVSHeader(string(raw))
	if err != nil {
		return nil, err
	}
	if metadata.rows > uint64(maximumBytes/32) {
		return nil, errors.New("DVS recording exceeds event budget")
	}
	values := int(metadata.rows) * 4
	maximumInt := int(^uint(0) >> 1)
	if values > maximumInt/metadata.dtype.width {
		return nil, errors.New("DVS source payload exceeds native size")
	}
	payload := make([]byte, values*metadata.dtype.width)
	if _, err = io.ReadFull(file, payload); err != nil {
		return nil, err
	}
	var extra [1]byte
	n, readErr := file.Read(extra[:])
	if n != 0 || readErr != io.EOF {
		return nil, errors.New("DVS recording contains extra content or unreadable tail")
	}
	events = make([]float64, values)
	for i := 0; i < values; i++ {
		value := metadata.dtype.decode(payload[i*metadata.dtype.width : (i+1)*metadata.dtype.width])
		destination := i
		if metadata.fortran {
			destination = (i%int(metadata.rows))*4 + i/int(metadata.rows)
		}
		events[destination] = value
	}
	return events, nil
}
