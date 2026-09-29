// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

package loaders

import (
	"strconv"
	"strings"
	"unicode/utf8"
)

// stringValue joins adjacent Python string literals without invoking an interpreter.
func (parser *dvsLiterals) stringValue() (any, error) {
	var output strings.Builder
	for {
		parser.whitespace()
		if parser.position == len(parser.text) {
			break
		}
		c := parser.text[parser.position]
		raw := false
		if c == 'r' || c == 'R' || c == 'u' || c == 'U' {
			if parser.position+1 == len(parser.text) || (parser.text[parser.position+1] != '\'' && parser.text[parser.position+1] != '"') {
				break
			}
			raw = c == 'r' || c == 'R'
			parser.position++
			c = parser.text[parser.position]
		} else if c != '\'' && c != '"' {
			break
		}
		delimiter := string(c)
		if strings.HasPrefix(parser.text[parser.position:], strings.Repeat(delimiter, 3)) {
			delimiter = strings.Repeat(delimiter, 3)
		}
		parser.position += len(delimiter)
		closed := false
		for parser.position < len(parser.text) {
			if strings.HasPrefix(parser.text[parser.position:], delimiter) {
				parser.position += len(delimiter)
				closed = true
				break
			}
			ch := parser.text[parser.position]
			parser.position++
			if ch == 0 || (ch == '\n' && len(delimiter) == 1) {
				return nil, errDVSLiteral
			}
			if ch != '\\' {
				output.WriteByte(ch)
				continue
			}
			if parser.position == len(parser.text) {
				return nil, errDVSLiteral
			}
			escaped := parser.text[parser.position]
			parser.position++
			if raw {
				output.WriteByte('\\')
				output.WriteByte(escaped)
				continue
			}
			switch escaped {
			case '\n':
			case '\\', '\'', '"':
				output.WriteByte(escaped)
			case 'a':
				output.WriteByte(7)
			case 'b':
				output.WriteByte(8)
			case 'f':
				output.WriteByte(12)
			case 'n':
				output.WriteByte(10)
			case 'r':
				output.WriteByte(13)
			case 't':
				output.WriteByte(9)
			case 'v':
				output.WriteByte(11)
			case 'N':
				if parser.position >= len(parser.text) || parser.text[parser.position] != '{' {
					return nil, errDVSLiteral
				}
				end := strings.IndexByte(parser.text[parser.position:], '}')
				if end < 0 {
					return nil, errDVSLiteral
				}
				name := strings.ToUpper(parser.text[parser.position+1 : parser.position+end])
				letter, ok := dvsNamedASCII(name)
				if !ok {
					return nil, errDVSLiteral
				}
				output.WriteByte(letter)
				parser.position += end + 1
			case 'x', 'u', 'U':
				width := 2
				if escaped == 'u' {
					width = 4
				}
				if escaped == 'U' {
					width = 8
				}
				if len(parser.text)-parser.position < width {
					return nil, errDVSLiteral
				}
				digits := parser.text[parser.position : parser.position+width]
				number, err := strconv.ParseUint(digits, 16, 32)
				if err != nil || number > utf8.MaxRune {
					return nil, errDVSLiteral
				}
				output.WriteRune(rune(number))
				parser.position += width
			default:
				if escaped >= '0' && escaped <= '7' {
					digits := string(escaped)
					for len(digits) < 3 && parser.position < len(parser.text) && parser.text[parser.position] >= '0' && parser.text[parser.position] <= '7' {
						digits += string(parser.text[parser.position])
						parser.position++
					}
					number, err := strconv.ParseUint(digits, 8, 16)
					if err != nil {
						return nil, errDVSLiteral
					}
					output.WriteRune(rune(number))
				} else {
					output.WriteByte('\\')
					output.WriteByte(escaped)
				}
			}
		}
		if !closed {
			return nil, errDVSLiteral
		}
	}
	return output.String(), nil
}

// dvsNamedASCII resolves named escapes that can occur in real scalar descriptors or keys.
func dvsNamedASCII(name string) (byte, bool) {
	punctuation := map[string]byte{"LESS-THAN SIGN": '<', "GREATER-THAN SIGN": '>', "EQUALS SIGN": '=', "VERTICAL LINE": '|', "PLUS SIGN": '+', "HYPHEN-MINUS": '-', "QUESTION MARK": '?', "LOW LINE": '_'}
	if character, ok := punctuation[name]; ok {
		return character, true
	}
	for _, prefix := range []string{"LATIN SMALL LETTER ", "LATIN CAPITAL LETTER "} {
		suffix := strings.TrimPrefix(name, prefix)
		if suffix != name && len(suffix) == 1 && suffix[0] >= 'A' && suffix[0] <= 'Z' {
			letter := suffix[0]
			if prefix == "LATIN SMALL LETTER " {
				letter += 'a' - 'A'
			}
			return letter, true
		}
	}
	for digit, word := range []string{"ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"} {
		if name == "DIGIT "+word {
			return byte('0' + digit), true
		}
	}
	return 0, false
}
