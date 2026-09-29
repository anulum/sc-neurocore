// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Converted DVS NPY recording

//! Scalar descriptor validation and native numeric conversion.
use crate::invalid;
use std::{io, mem};
unsafe extern "C" {
    fn sc_dvs_extended_width() -> usize;
    fn sc_dvs_long_width() -> usize;
    fn sc_dvs_extended(source: *const u8, big_endian: i32) -> f64;
}
pub(crate) struct Dtype {
    pub width: usize,
    kind: u8,
    big: bool,
}
impl Dtype {
    pub(crate) fn parse(source: &str) -> io::Result<Self> {
        let explicit = source.starts_with(['<', '>', '=', '|']);
        let (prefix, word) = if explicit {
            (source.as_bytes()[0], &source[1..])
        } else {
            (b'=', source)
        };
        let named = match word {
            "bool" | "bool_" => Some("b1"),
            "byte" => Some("i1"),
            "ubyte" => Some("u1"),
            "short" => Some("i2"),
            "ushort" => Some("u2"),
            "intc" | "int32" => Some("i4"),
            "uintc" | "uint32" => Some("u4"),
            "int8" => Some("i1"),
            "uint8" => Some("u1"),
            "int16" => Some("i2"),
            "uint16" => Some("u2"),
            "int64" | "longlong" => Some("i8"),
            "uint64" | "ulonglong" => Some("u8"),
            "half" | "float16" => Some("f2"),
            "single" | "float32" => Some("f4"),
            "double" | "float64" | "float" => Some("f8"),
            "longdouble" | "float128" => Some("f16"),
            _ => None,
        };
        if explicit
            && (named.is_some()
                || matches!(
                    word,
                    "int" | "int_" | "uint" | "intp" | "uintp" | "long" | "ulong"
                ))
        {
            return Err(invalid("prefixed named DVS dtype"));
        }
        // SAFETY: these no-argument C functions return host ABI sizes only.
        let long_width = unsafe { sc_dvs_long_width() };
        let pointer = mem::size_of::<usize>();
        let resolved = match word {
            "\u{0}" | "?" => "b1".into(),
            "\u{1}" => "i1".into(),
            "\u{2}" => "u1".into(),
            "\u{3}" => "i2".into(),
            "\u{4}" => "u2".into(),
            "\u{5}" => "i4".into(),
            "\u{6}" => "u4".into(),
            "\u{9}" => "i8".into(),
            "\u{a}" => "u8".into(),
            "\u{b}" => "f4".into(),
            "\u{c}" => "f8".into(),
            "\u{d}" => "f16".into(),
            "\u{17}" => "f2".into(),
            "b" => "i1".into(),
            "B" => "u1".into(),
            "h" => "i2".into(),
            "H" => "u2".into(),
            "i" => "i4".into(),
            "I" => "u4".into(),
            "q" => "i8".into(),
            "Q" => "u8".into(),
            "e" => "f2".into(),
            "f" => "f4".into(),
            "d" => "f8".into(),
            "g" => "f16".into(),
            "int" | "int_" | "intp" | "p" | "n" => format!("i{pointer}"),
            "uint" | "uintp" | "P" | "N" => format!("u{pointer}"),
            "\u{7}" | "long" | "l" => format!("i{long_width}"),
            "\u{8}" | "ulong" | "L" => format!("u{long_width}"),
            _ => named.unwrap_or(word).to_owned(),
        };
        if !resolved.is_ascii() || resolved.len() < 2 {
            return Err(invalid("invalid DVS scalar dtype"));
        }
        let kind = resolved.as_bytes()[0];
        let width = resolved[1..]
            .parse::<usize>()
            .map_err(|_| invalid("invalid DVS scalar width"))?;
        let valid = match kind {
            b'b' => width == 1,
            b'i' | b'u' => matches!(width, 1 | 2 | 4 | 8),
            b'f' => matches!(width, 2 | 4 | 8 | 16),
            _ => false,
        };
        if !valid {
            return Err(invalid("nonreal or incompatible DVS scalar"));
        }
        // SAFETY: the query has no pointers or side effects and reports the C ABI width.
        if width == 16 && unsafe { sc_dvs_extended_width() } != 16 {
            return Err(invalid("DVS extended ABI is unavailable"));
        }
        let big = match prefix {
            b'>' => true,
            b'<' => false,
            _ => cfg!(target_endian = "big"),
        };
        Ok(Self { width, kind, big })
    }
    pub(crate) fn decode(&self, raw: &[u8]) -> f64 {
        if self.kind == b'b' {
            return f64::from(raw[0] != 0);
        }
        if self.width == 16 {
            // SAFETY: the reader passes one live 16-byte scalar and parse checked the host ABI.
            return unsafe { sc_dvs_extended(raw.as_ptr(), i32::from(self.big)) };
        }
        let mut bits = 0u64;
        for (index, byte) in raw.iter().enumerate() {
            let shift = if self.big {
                self.width - 1 - index
            } else {
                index
            };
            bits |= u64::from(*byte) << (shift * 8)
        }
        match self.kind {
            b'u' => bits as f64,
            b'i' => {
                let shift = 64 - self.width * 8;
                ((bits << shift) as i64 >> shift) as f64
            }
            _ => match self.width {
                8 => f64::from_bits(bits),
                4 => f64::from(f32::from_bits(bits as u32)),
                _ => {
                    let exponent = ((bits >> 10) & 31) as i32;
                    let fraction = bits & 1023;
                    let value = if exponent == 31 {
                        if fraction == 0 {
                            f64::INFINITY
                        } else {
                            f64::NAN
                        }
                    } else if exponent == 0 {
                        (fraction as f64) * 2f64.powi(-24)
                    } else {
                        ((1024 + fraction) as f64) * 2f64.powi(exponent - 25)
                    };
                    if bits & 32768 != 0 {
                        -value
                    } else {
                        value
                    }
                }
            },
        }
    }
}
