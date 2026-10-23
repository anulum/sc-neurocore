// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Register transaction text parser

//! Strict register and store text grammar.

use super::{make_err, parse_type, parse_value_id, reserve_id, ParseError};
use crate::ir::graph::*;

pub(super) fn parse(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let op = instruction(text).map_err(|message| make_err(line, message))?;
    reserve_id(graph, op.result_id(), line)?;
    graph.push(op);
    Ok(())
}

fn instruction(text: &str) -> Result<ScOp, String> {
    let (result, body) = text.split_once(" = ").ok_or("expected state assignment")?;
    let id = parse_value_id(result)?;
    if let Some(body) = body.strip_prefix("sc.state ") {
        let (initial, ty) = body
            .split_once(" : ")
            .ok_or("State requires reset value and explicit type")?;
        return Ok(ScOp::State {
            id,
            initial: parse_value_id(initial)?,
            ty: parse_type(ty)?,
        });
    }
    let body = body.strip_prefix("sc.store ").ok_or("expected sc.store")?;
    let args: Vec<_> = body.split(',').map(str::trim).collect();
    if args.len() != 3 {
        return Err("Store requires state, next and enable=none or enable=%id".into());
    }
    let enable = args[2]
        .strip_prefix("enable=")
        .ok_or("Store requires explicit enable attribute")?;
    let enable = if enable == "none" {
        None
    } else {
        Some(parse_value_id(enable)?)
    };
    Ok(ScOp::Store {
        id,
        state: parse_value_id(args[0])?,
        next: parse_value_id(args[1])?,
        enable,
    })
}
