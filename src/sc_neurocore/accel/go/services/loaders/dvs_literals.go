// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

package loaders

import (
	"errors"
	"math/big"
	"regexp"
	"strings"
)

type dvsTuple []any
type dvsSigned struct{ number *big.Int }
type dvsLiterals struct {
	text     string
	position int
	depth    int
}

var dvsInteger = regexp.MustCompile(`^(?:0[xX](?:_?[0-9a-fA-F])+|0[oO](?:_?[0-7])+|0[bB](?:_?[01])+|0(?:_?0)*|[1-9](?:_?[0-9])*)$`)
var errDVSLiteral = errors.New("invalid DVS NPY literal")

// whitespace consumes Python spacing, comments and explicit line continuations.
func (parser *dvsLiterals) whitespace() {
	for parser.position < len(parser.text) {
		switch parser.text[parser.position] {
		case ' ', '\t', '\f', '\r', '\n':
			parser.position++
		case '#':
			for parser.position < len(parser.text) && parser.text[parser.position] != '\n' {
				parser.position++
			}
		case '\\':
			if strings.HasPrefix(parser.text[parser.position:], "\\\n") {
				parser.position += 2
			} else {
				return
			}
		default:
			return
		}
	}
}

func (parser *dvsLiterals) consume(token byte) bool {
	parser.whitespace()
	if parser.position < len(parser.text) && parser.text[parser.position] == token {
		parser.position++
		return true
	}
	return false
}

// value parses only inert scalar, tuple and dictionary literals, never expressions or calls.
func (parser *dvsLiterals) value() (any, error) {
	parser.whitespace()
	if parser.position == len(parser.text) {
		return nil, errDVSLiteral
	}
	token := parser.text[parser.position]
	if token == '(' || token == '{' {
		parser.depth++
		defer func() { parser.depth-- }()
		if parser.depth > 200 {
			return nil, errDVSLiteral
		}
	}
	if token == '\'' || token == '"' || ((token == 'u' || token == 'U' || token == 'r' || token == 'R') && parser.position+1 < len(parser.text) && (parser.text[parser.position+1] == '\'' || parser.text[parser.position+1] == '"')) {
		return parser.stringValue()
	}
	if token == '{' {
		return parser.dictionary()
	}
	if token == '(' {
		parser.position++
		if parser.consume(')') {
			return dvsTuple{}, nil
		}
		first, err := parser.value()
		if err != nil {
			return nil, err
		}
		if parser.consume(')') {
			return first, nil
		}
		if !parser.consume(',') {
			return nil, errDVSLiteral
		}
		values := dvsTuple{first}
		for !parser.consume(')') {
			value, err := parser.value()
			if err != nil {
				return nil, err
			}
			values = append(values, value)
			if parser.consume(')') {
				return values, nil
			}
			if !parser.consume(',') {
				return nil, errDVSLiteral
			}
		}
		return values, nil
	}
	sign := byte(0)
	if token == '+' || token == '-' {
		sign = token
		parser.position++
		parser.whitespace()
	}
	if sign != 0 && parser.position < len(parser.text) && parser.text[parser.position] == '(' {
		operand, err := parser.value()
		if err != nil {
			return nil, err
		}
		number, ok := operand.(*big.Int)
		if !ok {
			return nil, errDVSLiteral
		}
		if sign == '-' {
			number.Neg(number)
		}
		return dvsSigned{number}, nil
	}
	start := parser.position
	for parser.position < len(parser.text) {
		c := parser.text[parser.position]
		if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' {
			parser.position++
		} else {
			break
		}
	}
	word := parser.text[start:parser.position]
	if sign == 0 && word == "True" {
		return true, nil
	}
	if sign == 0 && word == "False" {
		return false, nil
	}
	if !dvsInteger.MatchString(word) {
		return nil, errDVSLiteral
	}
	cleaned := strings.ReplaceAll(word, "_", "")
	if !strings.HasPrefix(cleaned, "0x") && !strings.HasPrefix(cleaned, "0X") && !strings.HasPrefix(cleaned, "0o") && !strings.HasPrefix(cleaned, "0O") && !strings.HasPrefix(cleaned, "0b") && !strings.HasPrefix(cleaned, "0B") && len(cleaned) > 4300 {
		return nil, errDVSLiteral
	}
	number, ok := new(big.Int).SetString(cleaned, 0)
	if !ok {
		return nil, errDVSLiteral
	}
	if sign == '-' {
		number.Neg(number)
	}
	if sign != 0 {
		return dvsSigned{number}, nil
	}
	return number, nil
}

func (parser *dvsLiterals) dictionary() (any, error) {
	parser.position++
	result := make(map[string]any)
	if parser.consume('}') {
		return result, nil
	}
	for {
		keyValue, err := parser.value()
		if err != nil {
			return nil, err
		}
		key, ok := keyValue.(string)
		if !ok {
			return nil, errDVSLiteral
		}
		if _, exists := result[key]; exists {
			return nil, errDVSLiteral
		}
		if !parser.consume(':') {
			return nil, errDVSLiteral
		}
		value, err := parser.value()
		if err != nil {
			return nil, err
		}
		result[key] = value
		if parser.consume('}') {
			return result, nil
		}
		if !parser.consume(',') {
			return nil, errDVSLiteral
		}
		if parser.consume('}') {
			return result, nil
		}
	}
}
