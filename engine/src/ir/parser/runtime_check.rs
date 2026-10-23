// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Runtime check text parser

//! Preserve explicit check reason and Boolean control without producing a value.

use super::{make_err, parse_value_id, reserve_id, ParseError};
use crate::ir::graph::*;

pub(super) fn parse(text: &str, graph: &mut ScGraph, line: usize) -> Result<(), ParseError> {
    let op = instruction(text).map_err(|message| make_err(line, message))?;
    reserve_id(graph, op.result_id(), line)?;
    graph.push(op);
    Ok(())
}

fn instruction(text: &str) -> Result<ScOp, String> {
    let (result, body) = text.split_once(" = ").ok_or("expected check assignment")?;
    let body = body.strip_prefix("sc.check ").ok_or("expected sc.check")?;
    let args: Vec<_> = body.split(',').map(str::trim).collect();
    let [kind, condition, enable] = args.as_slice() else {
        return Err("Check requires kind, condition and enable=none or enable=%id".into());
    };
    let enable = enable
        .strip_prefix("enable=")
        .ok_or("Check requires an explicit enable attribute")?;
    Ok(ScOp::Check {
        id: parse_value_id(result)?,
        kind: kind.parse()?,
        condition: parse_value_id(condition)?,
        enable: if enable == "none" {
            None
        } else {
            Some(parse_value_id(enable)?)
        },
    })
}
