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
	"runtime"
	"strconv"
	"strings"
)

var dvsDtypeAliases = map[string]string{
	"\x00": "b1", "\x01": "i1", "\x02": "u1", "\x03": "i2", "\x04": "u2",
	"\x05": "i4", "\x06": "u4", "\x09": "i8", "\x0a": "u8", "\x0b": "f4",
	"\x0c": "f8", "\x0d": "f16", "\x17": "f2",
	"bool":       "b1",
	"bool_":      "b1",
	"?":          "b1",
	"byte":       "i1",
	"ubyte":      "u1",
	"short":      "i2",
	"ushort":     "u2",
	"intc":       "i4",
	"uintc":      "u4",
	"longlong":   "i8",
	"ulonglong":  "u8",
	"int64":      "i8",
	"uint64":     "u8",
	"int32":      "i4",
	"uint32":     "u4",
	"int16":      "i2",
	"uint16":     "u2",
	"int8":       "i1",
	"uint8":      "u1",
	"half":       "f2",
	"single":     "f4",
	"double":     "f8",
	"float16":    "f2",
	"float32":    "f4",
	"float64":    "f8",
	"float":      "f8",
	"longdouble": "f16",
	"float128":   "f16",
	"b":          "i1",
	"B":          "u1",
	"h":          "i2",
	"H":          "u2",
	"i":          "i4",
	"I":          "u4",
	"q":          "i8",
	"Q":          "u8",
	"e":          "f2",
	"f":          "f4",
	"d":          "f8",
	"g":          "f16",
}

// parseDVSDtype keeps NumPy scalar aliases and platform-dependent native widths explicit.
func parseDVSDtype(descriptor string) (dvsDtype, error) {
	result := dvsDtype{}
	fail := errors.New("unsupported or nonreal DVS NPY scalar dtype")
	nativeBig := binary.NativeEndian.Uint16([]byte{0, 1}) == 1
	result.bigEndian = nativeBig
	prefix := byte('=')
	explicit := false
	if descriptor != "" && strings.ContainsRune("<>=|", rune(descriptor[0])) {
		explicit = true
		prefix = descriptor[0]
		descriptor = descriptor[1:]
	}
	switch prefix {
	case '<':
		result.bigEndian = false
	case '>':
		result.bigEndian = true
	}
	// Character codes accept byte-order prefixes. Named scalar aliases do not.
	if len(descriptor) > 1 && explicit && descriptor[0] != 'b' && descriptor[0] != 'i' && descriptor[0] != 'u' && descriptor[0] != 'f' {
		return result, fail
	}
	if explicit && len(descriptor) > 1 {
		if _, alias := dvsDtypeAliases[descriptor]; alias {
			return result, fail
		}
	}
	pointer := strconv.Itoa(strconv.IntSize / 8)
	longWidth := pointer
	if runtime.GOOS == "windows" {
		longWidth = "4"
	}
	switch descriptor {
	case "int", "int_", "intp", "p", "n":
		descriptor = "i" + pointer
	case "uint", "uintp", "P", "N":
		descriptor = "u" + pointer
	case "\x07", "long", "l":
		descriptor = "i" + longWidth
	case "\x08", "ulong", "L":
		descriptor = "u" + longWidth
	default:
		if alias, ok := dvsDtypeAliases[descriptor]; ok {
			descriptor = alias
		}
	}
	if len(descriptor) < 2 {
		return result, fail
	}
	result.kind = descriptor[0]
	width, err := strconv.Atoi(descriptor[1:])
	if err != nil {
		return result, fail
	}
	result.width = width
	switch result.kind {
	case 'b':
		if width != 1 {
			return result, fail
		}
	case 'u', 'i':
		if width != 1 && width != 2 && width != 4 && width != 8 {
			return result, fail
		}
	case 'f':
		if width != 2 && width != 4 && width != 8 && width != 16 {
			return result, fail
		}
		if width == 16 && runtime.GOARCH != "amd64" {
			return result, errors.New("DVS extended float representation requires x86-64 host")
		}
	default:
		return result, fail
	}
	return result, nil
}
