// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

//! Exact NPY metadata validation after inert literal parsing.
use crate::{
    dtype::Dtype,
    invalid,
    literals::{Node, Parser},
};
use std::io;
pub(crate) struct Header {
    pub rows: usize,
    pub fortran: bool,
    pub dtype: Dtype,
}
/// Validate the three unique metadata fields and the two-dimensional shape.
pub(crate) fn parse(text: &str) -> io::Result<Header> {
    if text.contains('\0') {
        return Err(invalid("NUL in DVS header"));
    }
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut parser = Parser::new(&normalized);
    parser.whitespace();
    let prefix = &normalized[..parser.position];
    let indentation = prefix
        .rsplit('\n')
        .next()
        .unwrap_or("")
        .rsplit('\u{c}')
        .next()
        .unwrap_or("");
    if !indentation.is_empty() && indentation.chars().all(|c| c == ' ' || c == '\t') {
        return Err(invalid("indented DVS header"));
    }
    let root = parser.value()?;
    parser.whitespace();
    if parser.position != normalized.len() {
        return Err(invalid("trailing DVS expression"));
    }
    let Node::Dict(fields) = root else {
        return Err(invalid("DVS header is not a dictionary"));
    };
    if fields.len() != 3 {
        return Err(invalid("unexpected DVS header fields"));
    }
    let mut descriptor = None;
    let mut fortran = None;
    let mut rows = None;
    for (key, value) in fields {
        match (key.as_str(), value) {
            ("descr", Node::String(text)) => descriptor = Some(text),
            ("fortran_order", Node::Bool(value)) => fortran = Some(value),
            ("shape", Node::Tuple(values)) if values.len() == 2 => {
                let mut items = values.into_iter();
                let row = items.next().ok_or_else(|| invalid("missing DVS rows"))?;
                let column = items.next().ok_or_else(|| invalid("missing DVS columns"))?;
                let Node::Number {
                    value: Some(n),
                    negative,
                    ..
                } = row
                else {
                    return Err(invalid("invalid DVS rows"));
                };
                if negative && n != 0 {
                    return Err(invalid("negative DVS rows"));
                }
                if !matches!(
                    column,
                    Node::Number {
                        value: Some(4),
                        negative: false,
                        ..
                    }
                ) {
                    return Err(invalid("DVS requires four columns"));
                }
                rows =
                    Some(usize::try_from(n).map_err(|_| invalid("DVS rows exceed native size"))?);
            }
            _ => return Err(invalid("invalid DVS header field")),
        }
    }
    Ok(Header {
        rows: rows.ok_or_else(|| invalid("missing DVS shape"))?,
        fortran: fortran.ok_or_else(|| invalid("missing DVS order"))?,
        dtype: Dtype::parse(&descriptor.ok_or_else(|| invalid("missing DVS dtype"))?)?,
    })
}
