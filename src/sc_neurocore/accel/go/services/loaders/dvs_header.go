// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

package loaders

import (
	"math/big"
	"strings"
)

type dvsHeader struct {
	rows    uint64
	fortran bool
	dtype   dvsDtype
}

// parseDVSHeader validates inert dictionary values, including grouped/escaped literals.
func parseDVSHeader(raw string) (dvsHeader, error) {
	result := dvsHeader{}
	if strings.ContainsRune(raw, 0) {
		return result, errDVSLiteral
	}
	parser := dvsLiterals{text: strings.ReplaceAll(strings.ReplaceAll(raw, "\r\n", "\n"), "\r", "\n")}
	parser.whitespace()
	lineStart := strings.LastIndex(parser.text[:parser.position], "\n") + 1
	indentation := parser.text[lineStart:parser.position]
	if lastFormfeed := strings.LastIndex(indentation, "\f"); lastFormfeed >= 0 {
		indentation = indentation[lastFormfeed+1:]
	}
	if strings.Trim(indentation, " \t") == "" && len(indentation) != 0 {
		return result, errDVSLiteral
	}
	value, err := parser.value()
	if err != nil {
		return result, err
	}
	parser.whitespace()
	if parser.position != len(parser.text) {
		return result, errDVSLiteral
	}
	fields, ok := value.(map[string]any)
	if !ok || len(fields) != 3 {
		return result, errDVSLiteral
	}
	descriptor, ok := fields["descr"].(string)
	if !ok {
		return result, errDVSLiteral
	}
	fortran, ok := fields["fortran_order"].(bool)
	if !ok {
		return result, errDVSLiteral
	}
	shape, ok := fields["shape"].(dvsTuple)
	if !ok || len(shape) != 2 {
		return result, errDVSLiteral
	}
	rows, ok := dvsIntegerValue(shape[0])
	if !ok || rows.Sign() < 0 || !rows.IsUint64() {
		return result, errDVSLiteral
	}
	columns, ok := dvsIntegerValue(shape[1])
	if !ok || columns.Cmp(big.NewInt(4)) != 0 {
		return result, errDVSLiteral
	}
	dtype, err := parseDVSDtype(descriptor)
	if err != nil {
		return result, err
	}
	result.rows = rows.Uint64()
	result.fortran = fortran
	result.dtype = dtype
	return result, nil
}

// dvsIntegerValue distinguishes signed integers from Boolean literals.
func dvsIntegerValue(value any) (*big.Int, bool) {
	switch integer := value.(type) {
	case *big.Int:
		return integer, true
	case dvsSigned:
		return integer.number, true
	default:
		return nil, false
	}
}
