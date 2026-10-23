// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scalar instruction text parser

use super::{make_err, parse_type, parse_value_id, ParseError};
use crate::ir::{graph::*, word::*};

pub(super) fn parse(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let result = instruction(text).map_err(|e| make_err(line, e))?;
    graph.next_id = graph.next_id.max(
        result
            .0
             .0
            .checked_add(1)
            .ok_or_else(|| make_err(line, "value ID overflow"))?,
    );
    graph.push(ScOp::Word {
        id: result.0,
        op: result.1,
    });
    Ok(())
}

fn instruction(text: &str) -> Result<(ValueId, WordOp), String> {
    let (id, body) = text
        .split_once(" = sc.word ")
        .ok_or("expected scalar word assignment")?;
    let id = parse_value_id(id)?;
    let (body, ty) = body
        .rsplit_once(" : ")
        .ok_or("word operation needs an explicit result type")?;
    let ty = parse_type(ty)?;
    let (kind, args) = body
        .split_once(' ')
        .ok_or("word operation needs operands")?;
    let args: Vec<_> = args.split(',').map(str::trim).collect();
    let count = |n| {
        if args.len() == n {
            Ok(())
        } else {
            Err(format!("{kind} requires {n} arguments, got {}", args.len()))
        }
    };
    let op = match kind {
        "delay" => {
            count(1)?;
            WordOp::Delay {
                input: parse_value_id(args[0])?,
                ty,
            }
        }
        "resize" | "bitcast" => {
            count(1)?;
            let input = parse_value_id(args[0])?;
            if kind == "resize" {
                WordOp::Resize { input, ty }
            } else {
                WordOp::Bitcast { input, ty }
            }
        }
        "select" => {
            count(3)?;
            WordOp::Select {
                condition: parse_value_id(args[0])?,
                when_true: parse_value_id(args[1])?,
                when_false: parse_value_id(args[2])?,
                ty,
            }
        }
        "round" => {
            count(3)?;
            WordOp::Round {
                input: parse_value_id(args[0])?,
                shift: args[1].parse::<u32>().map_err(|e| e.to_string())?,
                mode: args[2].parse()?,
                ty,
            }
        }
        "clamp" => {
            count(3)?;
            WordOp::Clamp {
                input: parse_value_id(args[0])?,
                min: args[1].parse::<i64>().map_err(|e| e.to_string())?,
                max: args[2].parse::<i64>().map_err(|e| e.to_string())?,
                ty,
            }
        }
        "neg" | "not" => {
            count(1)?;
            WordOp::Unary {
                kind: kind.parse()?,
                input: parse_value_id(args[0])?,
                ty,
            }
        }
        _ if kind.starts_with("cmp_") => {
            count(2)?;
            if ty != ScType::Bool {
                return Err("word comparison result must be bool".into());
            }
            WordOp::Compare {
                predicate: kind[4..].parse()?,
                lhs: parse_value_id(args[0])?,
                rhs: parse_value_id(args[1])?,
            }
        }
        _ => {
            count(2)?;
            WordOp::Binary {
                kind: kind.parse()?,
                lhs: parse_value_id(args[0])?,
                rhs: parse_value_id(args[1])?,
                ty,
            }
        }
    };
    Ok((id, op))
}
