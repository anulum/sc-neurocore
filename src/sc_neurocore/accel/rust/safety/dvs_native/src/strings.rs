// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

//! Python string literal decoding for exact metadata keys and real dtypes.
use crate::{invalid, literals::Parser};
use std::io;
impl Parser<'_> {
    pub fn string_start(&self) -> bool {
        matches!(self.peek(), Some(b'\'' | b'"'))
            || (matches!(self.peek(), Some(b'r' | b'R' | b'u' | b'U'))
                && matches!(
                    self.text.as_bytes().get(self.position + 1),
                    Some(b'\'' | b'"')
                ))
    }
    /// Join adjacent literals, preserving raw escapes and Unicode boundaries.
    pub fn string_value(&mut self) -> io::Result<String> {
        let mut result = String::new();
        loop {
            self.whitespace();
            if !self.string_start() {
                break;
            }
            let raw = matches!(self.peek(), Some(b'r' | b'R'));
            if matches!(self.peek(), Some(b'r' | b'R' | b'u' | b'U')) {
                self.position += 1
            }
            let quote = self
                .peek()
                .ok_or_else(|| invalid("missing DVS string quote"))?;
            let triple = self.text.as_bytes()[self.position..].starts_with(&[quote; 3]);
            let width = if triple { 3 } else { 1 };
            self.position += width;
            loop {
                if self.text.as_bytes()[self.position..].starts_with(&[quote; 3][..width]) {
                    self.position += width;
                    break;
                }
                let ch = self.text[self.position..]
                    .chars()
                    .next()
                    .ok_or_else(|| invalid("unclosed DVS string"))?;
                self.position += ch.len_utf8();
                if ch == '\n' && !triple {
                    return Err(invalid("newline in DVS string"));
                }
                if ch != '\\' {
                    result.push(ch);
                    continue;
                }
                let escaped = self.text[self.position..]
                    .chars()
                    .next()
                    .ok_or_else(|| invalid("unterminated DVS escape"))?;
                self.position += escaped.len_utf8();
                if raw {
                    result.push('\\');
                    result.push(escaped)
                } else {
                    self.escape(escaped, &mut result)?
                }
            }
        }
        Ok(result)
    }
    fn escape(&mut self, ch: char, result: &mut String) -> io::Result<()> {
        match ch {
            '\n' => {}
            '\\' | '\'' | '"' => result.push(ch),
            'a' => result.push('\u{7}'),
            'b' => result.push('\u{8}'),
            'f' => result.push('\u{c}'),
            'n' => result.push('\n'),
            'r' => result.push('\r'),
            't' => result.push('\t'),
            'v' => result.push('\u{b}'),
            'x' | 'u' | 'U' => {
                let width = match ch {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                let end = self.position + width;
                let digits = self
                    .text
                    .get(self.position..end)
                    .ok_or_else(|| invalid("truncated DVS hexadecimal escape"))?;
                if !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(invalid("invalid DVS hexadecimal escape"));
                }
                let number = u32::from_str_radix(digits, 16)
                    .map_err(|_| invalid("invalid DVS codepoint"))?;
                result.push(
                    char::from_u32(number)
                        .ok_or_else(|| invalid("unsupported DVS Unicode scalar"))?,
                );
                self.position = end;
            }
            'N' => {
                if self.peek() != Some(b'{') {
                    return Err(invalid("invalid DVS named escape"));
                }
                self.position += 1;
                let suffix = &self.text[self.position..];
                let end = suffix
                    .find('}')
                    .ok_or_else(|| invalid("unclosed DVS named escape"))?;
                result.push(
                    named_ascii(&suffix[..end])
                        .ok_or_else(|| invalid("unsupported DVS named escape"))?,
                );
                self.position += end + 1;
            }
            '0'..='7' => {
                let mut number = u32::from(ch) - u32::from('0');
                for _ in 0..2 {
                    if let Some(byte @ b'0'..=b'7') = self.peek() {
                        number = number * 8 + u32::from(byte - b'0');
                        self.position += 1
                    } else {
                        break;
                    }
                }
                result.push(
                    char::from_u32(number).ok_or_else(|| invalid("invalid DVS octal escape"))?,
                );
            }
            _ => {
                result.push('\\');
                result.push(ch)
            }
        }
        Ok(())
    }
}
/// Named escapes capable of forming admitted ASCII field names and descriptors.
fn named_ascii(name: &str) -> Option<char> {
    let uppercase = name.to_ascii_uppercase();
    let name = uppercase.as_str();
    match name {
        "LESS-THAN SIGN" => return Some('<'),
        "GREATER-THAN SIGN" => return Some('>'),
        "EQUALS SIGN" => return Some('='),
        "VERTICAL LINE" => return Some('|'),
        "PLUS SIGN" => return Some('+'),
        "HYPHEN-MINUS" => return Some('-'),
        "QUESTION MARK" => return Some('?'),
        "LOW LINE" => return Some('_'),
        _ => {}
    }
    for (prefix, lower) in [
        ("LATIN SMALL LETTER ", true),
        ("LATIN CAPITAL LETTER ", false),
    ] {
        if let Some(suffix) = name.strip_prefix(prefix) {
            if suffix.len() == 1 && suffix.as_bytes()[0].is_ascii_uppercase() {
                let letter = char::from(suffix.as_bytes()[0]);
                return Some(if lower {
                    letter.to_ascii_lowercase()
                } else {
                    letter
                });
            }
        }
    }
    for (digit, word) in [
        "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
    ]
    .iter()
    .enumerate()
    {
        if name.strip_prefix("DIGIT ") == Some(*word) {
            return char::from_u32(u32::from('0') + digit as u32);
        }
    }
    None
}
