// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Effectful simulation checks

//! Typed simulation failures, excluded from synthesis without changing data words.

#![warn(missing_docs)]

use super::{graph::*, stream_contract::value_type, verify::VerifyError};
use std::{fmt, str::FromStr};

/// Failure category retained in graph text and emitted simulation diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeCheckKind {
    /// A declared encoded-integer range was exceeded.
    Overflow,
    /// An arithmetic precondition was not satisfied.
    ArithmeticDomain,
}

impl fmt::Display for RuntimeCheckKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Overflow => "overflow",
            Self::ArithmeticDomain => "domain",
        })
    }
}

impl FromStr for RuntimeCheckKind {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "overflow" => Ok(Self::Overflow),
            "domain" => Ok(Self::ArithmeticDomain),
            _ => Err(format!("unknown runtime check kind: {text:?}")),
        }
    }
}

impl RuntimeCheckKind {
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::Overflow => "SCIR overflow",
            Self::ArithmeticDomain => "SCIR arithmetic domain",
        }
    }
}

pub(crate) fn verify(graph: &ScGraph, errors: &mut Vec<VerifyError>) {
    for (op_index, op) in graph.ops.iter().enumerate() {
        let ScOp::Check {
            condition, enable, ..
        } = op
        else {
            continue;
        };
        for (id, name) in std::iter::once((*condition, "condition"))
            .chain(enable.iter().map(|id| (*id, "enable")))
        {
            let message = match value_type(graph, id) {
                Ok(ScType::Bool) => continue,
                Ok(ty) => format!("sc.check {name} requires bool, got {ty}"),
                Err(error) => format!("sc.check {name}: {error}"),
            };
            errors.push(VerifyError { op_index, message });
        }
    }
}
