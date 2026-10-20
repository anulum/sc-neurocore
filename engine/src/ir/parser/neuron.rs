// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compound neuron text decoding

//! Decode complete hardware parameters of compound neuron operations.

use std::collections::HashMap;
use std::str::FromStr;

use super::{make_err, parse_value_id, ParseError};
use crate::ir::graph::*;

struct Fields<'a> {
    values: HashMap<&'a str, &'a str>,
    line: usize,
}

impl<'a> Fields<'a> {
    fn new(text: &'a str, line: usize, allowed: &[&str]) -> Result<Self, ParseError> {
        let mut values = HashMap::new();
        for field in text.split(',') {
            let (key, value) = field
                .trim()
                .split_once('=')
                .ok_or_else(|| make_err(line, format!("expected key=value, got '{field}'")))?;
            let key = key.trim();
            if !allowed.contains(&key) {
                return Err(make_err(line, format!("unknown parameter '{key}'")));
            }
            if values.insert(key, value.trim()).is_some() {
                return Err(make_err(line, format!("duplicate parameter '{key}'")));
            }
        }
        Ok(Self { values, line })
    }

    fn required(&self, key: &str) -> Result<&str, ParseError> {
        self.values
            .get(key)
            .copied()
            .ok_or_else(|| make_err(self.line, format!("missing {key}")))
    }

    fn number<T: FromStr>(&self, key: &str, default: T) -> Result<T, ParseError> {
        match self.values.get(key) {
            None => Ok(default),
            Some(value) => value
                .parse()
                .map_err(|_| make_err(self.line, format!("invalid {key}='{value}'"))),
        }
    }

    fn operand(&self, key: &str) -> Result<ValueId, ParseError> {
        parse_value_id(self.required(key)?).map_err(|e| make_err(self.line, e))
    }

    fn offsets(&self) -> Result<Vec<u32>, ParseError> {
        let text = self.required("offsets")?;
        let inner = text
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| make_err(self.line, "offsets must be a bracketed list"))?;
        if inner.is_empty() {
            return Ok(Vec::new());
        }
        inner
            .split(';')
            .map(|value| {
                value
                    .trim()
                    .parse()
                    .map_err(|_| make_err(self.line, format!("invalid tap offset '{value}'")))
            })
            .collect()
    }
}

/// Parse a compound neuron operation without discarding hardware parameters.
pub(super) fn parse(
    text: &str,
    graph: &mut ScGraph,
    line: usize,
    name: &str,
) -> Result<(), ParseError> {
    let (id, rest) = text
        .split_once(&format!("= {name}"))
        .expect("compound operation dispatch matches the exact operation name");
    let id = parse_value_id(id).map_err(|e| make_err(line, e))?;
    let (body, _) = rest
        .split_once(':')
        .ok_or_else(|| make_err(line, "missing result type"))?;
    let (input, attributes) = body
        .split_once(',')
        .ok_or_else(|| make_err(line, "missing neuron operands"))?;
    let input = parse_value_id(input).map_err(|e| make_err(line, e))?;

    let op = match name {
        "sc.lif_step" => {
            let f = Fields::new(
                attributes,
                line,
                &[
                    "leak", "gain", "noise", "dw", "frac", "vr", "reset", "vt", "rp",
                ],
            )?;
            let d = LifParams::default();
            ScOp::LifStep {
                id,
                current: input,
                leak: f.operand("leak")?,
                gain: f.operand("gain")?,
                noise: f.operand("noise")?,
                params: LifParams {
                    data_width: f.number("dw", d.data_width)?,
                    fraction: f.number("frac", d.fraction)?,
                    v_rest: f.number("vr", d.v_rest)?,
                    v_reset: f.number("reset", d.v_reset)?,
                    v_threshold: f.number("vt", d.v_threshold)?,
                    refractory_period: f.number("rp", d.refractory_period)?,
                },
            }
        }
        "sc.dense_forward" => {
            let f = Fields::new(
                attributes,
                line,
                &[
                    "weights",
                    "leak",
                    "gain",
                    "ni",
                    "nn",
                    "len",
                    "dw",
                    "input_seed",
                    "weight_seed",
                    "y_min",
                    "y_max",
                ],
            )?;
            let d = DenseParams::default();
            ScOp::DenseForward {
                id,
                inputs: input,
                weights: f.operand("weights")?,
                leak: f.operand("leak")?,
                gain: f.operand("gain")?,
                params: DenseParams {
                    n_inputs: f.number("ni", d.n_inputs)?,
                    n_neurons: f.number("nn", d.n_neurons)?,
                    data_width: f.number("dw", d.data_width)?,
                    stream_length: f.number("len", d.stream_length)?,
                    input_seed_base: f.number("input_seed", d.input_seed_base)?,
                    weight_seed_base: f.number("weight_seed", d.weight_seed_base)?,
                    y_min: f.number("y_min", d.y_min)?,
                    y_max: f.number("y_max", d.y_max)?,
                },
            }
        }
        _ => {
            let f = Fields::new(
                attributes,
                line,
                &[
                    "weights", "centre", "sigma", "taps", "depth", "dw", "frac", "ptr", "offsets",
                ],
            )?;
            let d = DclsParams::default();
            ScOp::DclsLayer {
                id,
                spike: input,
                weights: f.operand("weights")?,
                centre: f.operand("centre")?,
                sigma: f.operand("sigma")?,
                params: DclsParams {
                    n_taps: f.number("taps", d.n_taps)?,
                    data_width: f.number("dw", d.data_width)?,
                    fraction: f.number("frac", d.fraction)?,
                    delay_depth: f.number("depth", d.delay_depth)?,
                    ptr_width: f.number("ptr", d.ptr_width)?,
                    tap_offsets: f.offsets()?,
                },
            }
        }
    };
    graph.next_id =
        graph.next_id.max(id.0.checked_add(1).ok_or_else(|| {
            make_err(line, "neuron result ID leaves no space for the next value")
        })?);
    graph.push(op);
    Ok(())
}
