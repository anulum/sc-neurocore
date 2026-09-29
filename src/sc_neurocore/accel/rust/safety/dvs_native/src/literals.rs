// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

//! Bounded inert Python literals for NPY headers; expressions never execute.
use crate::invalid;
use std::io;
pub(crate) enum Node {
    String(String),
    Bool(bool),
    Number {
        value: Option<u64>,
        negative: bool,
        signed: bool,
    },
    Tuple(Vec<Node>),
    Dict(Vec<(String, Node)>),
}
pub(crate) struct Parser<'a> {
    pub text: &'a str,
    pub position: usize,
    depth: usize,
}
impl<'a> Parser<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            position: 0,
            depth: 0,
        }
    }
    pub fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.position).copied()
    }
    /// Consume comments, Python whitespace and explicit line continuations.
    pub fn whitespace(&mut self) {
        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\n' | b'\r' | 12) => self.position += 1,
                Some(b'#') => {
                    while self.peek().is_some_and(|b| b != b'\n') {
                        self.position += 1
                    }
                }
                Some(b'\\') if self.text[self.position..].starts_with("\\\n") => self.position += 2,
                _ => break,
            }
        }
    }
    pub fn consume(&mut self, token: u8) -> bool {
        self.whitespace();
        if self.peek() == Some(token) {
            self.position += 1;
            true
        } else {
            false
        }
    }
    /// Parse one scalar or structurally bounded container.
    pub fn value(&mut self) -> io::Result<Node> {
        self.whitespace();
        let token = self.peek().ok_or_else(|| invalid("missing DVS literal"))?;
        if self.string_start() {
            return self.string_value().map(Node::String);
        }
        if matches!(token, b'(' | b'{') {
            self.depth += 1;
            if self.depth > 200 {
                return Err(invalid("DVS nesting exceeds Python parser limit"));
            }
            let result = if token == b'{' {
                self.dictionary()
            } else {
                self.tuple()
            };
            self.depth -= 1;
            return result;
        }
        let sign = if matches!(token, b'+' | b'-') {
            self.position += 1;
            self.whitespace();
            Some(token)
        } else {
            None
        };
        if sign.is_some() && self.peek() == Some(b'(') {
            let operand = self.value()?;
            let Node::Number {
                value,
                negative: false,
                signed: false,
            } = operand
            else {
                return Err(invalid("invalid DVS unary operand"));
            };
            return Ok(Node::Number {
                value,
                negative: sign == Some(b'-'),
                signed: true,
            });
        }
        let start = self.position;
        while self
            .peek()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            self.position += 1
        }
        let word = &self.text[start..self.position];
        if sign.is_none() {
            match word {
                "True" => return Ok(Node::Bool(true)),
                "False" => return Ok(Node::Bool(false)),
                _ => {}
            }
        }
        let value = integer(word)?;
        Ok(Node::Number {
            value,
            negative: sign == Some(b'-'),
            signed: sign.is_some(),
        })
    }
    fn tuple(&mut self) -> io::Result<Node> {
        self.position += 1;
        if self.consume(b')') {
            return Ok(Node::Tuple(Vec::new()));
        }
        let first = self.value()?;
        if self.consume(b')') {
            return Ok(first);
        }
        if !self.consume(b',') {
            return Err(invalid("missing DVS tuple comma"));
        }
        let mut values = vec![first];
        while !self.consume(b')') {
            values.push(self.value()?);
            if self.consume(b')') {
                break;
            }
            if !self.consume(b',') {
                return Err(invalid("missing DVS tuple separator"));
            }
        }
        Ok(Node::Tuple(values))
    }
    fn dictionary(&mut self) -> io::Result<Node> {
        self.position += 1;
        let mut values = Vec::new();
        if self.consume(b'}') {
            return Ok(Node::Dict(values));
        }
        loop {
            let Node::String(key) = self.value()? else {
                return Err(invalid("nonstring DVS dictionary key"));
            };
            if values.iter().any(|(name, _)| name == &key) {
                return Err(invalid("duplicate DVS dictionary key"));
            }
            if !self.consume(b':') {
                return Err(invalid("missing DVS field colon"));
            }
            values.push((key, self.value()?));
            if self.consume(b'}') {
                break;
            }
            if !self.consume(b',') {
                return Err(invalid("missing DVS dictionary separator"));
            }
            if self.consume(b'}') {
                break;
            }
        }
        Ok(Node::Dict(values))
    }
}
/// Validate Python integer spelling before detecting native-budget overflow.
fn integer(word: &str) -> io::Result<Option<u64>> {
    let (radix, digits, prefixed) = if word.starts_with("0x") || word.starts_with("0X") {
        (16, &word[2..], true)
    } else if word.starts_with("0o") || word.starts_with("0O") {
        (8, &word[2..], true)
    } else if word.starts_with("0b") || word.starts_with("0B") {
        (2, &word[2..], true)
    } else {
        (10, word, false)
    };
    if digits.is_empty() {
        return Err(invalid("empty DVS integer"));
    }
    let mut count = 0;
    let mut underscore = false;
    let mut value = Some(0u64);
    let mut nonzero = false;
    for (index, byte) in digits.bytes().enumerate() {
        if byte == b'_' {
            if underscore || (index == 0 && !prefixed) {
                return Err(invalid("invalid DVS integer underscore"));
            }
            underscore = true;
            continue;
        }
        let digit = char::from(byte)
            .to_digit(radix)
            .ok_or_else(|| invalid("invalid DVS integer digit"))?;
        nonzero |= digit != 0;
        count += 1;
        underscore = false;
        value = value
            .and_then(|v| v.checked_mul(u64::from(radix)))
            .and_then(|v| v.checked_add(u64::from(digit)));
    }
    if underscore
        || count == 0
        || (!prefixed && (count > 4300 || (digits.starts_with('0') && nonzero)))
    {
        return Err(invalid("invalid DVS integer spelling"));
    }
    Ok(value)
}
