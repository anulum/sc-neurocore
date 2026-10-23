// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Shared HDL identifier admission

//! Both hardware backends use the same public names and generated namespaces.

use super::{graph::*, verify::VerifyError};
use std::collections::HashSet;

const KEYWORDS: &str = "accept_on alias always always_comb always_ff always_latch and assert assign assume automatic before begin bind bins binsof bit break buf bufif0 bufif1 byte case casex casez cell chandle checker class clocking cmos config const constraint context continue cover covergroup coverpoint cross deassign default defparam design disable dist do edge else end endcase endchecker endclass endclocking endconfig endfunction endgenerate endgroup endinterface endmodule endpackage endprimitive endprogram endproperty endspecify endsequence endtable endtask enum event eventually expect export extends extern final first_match for force foreach forever fork forkjoin function generate genvar global highz0 highz1 if iff ifnone ignore_bins illegal_bins implements implies import incdir include initial inout input inside instance int integer interconnect interface intersect join join_any join_none large let liblist library local localparam logic longint macromodule matches medium modport module nand negedge nettype new nexttime nmos nor noshowcancelled not notif0 notif1 null or output package packed parameter pmos posedge primitive priority program property protected pull0 pull1 pulldown pullup pulsestyle_ondetect pulsestyle_onevent pure rand randc randcase randsequence rcmos real realtime ref reg reject_on release repeat restrict return rnmos rpmos rtran rtranif0 rtranif1 s_always s_eventually s_nexttime s_until s_until_with scalared sequence shortint shortreal showcancelled signed small soft solve specify specparam static string strong strong0 strong1 struct super supply0 supply1 sync_accept_on sync_reject_on table tagged task this throughout time timeprecision timeunit tran tranif0 tranif1 tri tri0 tri1 triand trior trireg type typedef union unique unique0 unsigned until until_with untyped use uwire var vectored virtual void wait wait_order wand weak weak0 weak1 while wildcard wire with within wor xnor xor";

fn valid(name: &str) -> bool {
    let mut chars = name.bytes();
    matches!(chars.next(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'_'))
        && chars.all(|c| c.is_ascii_alphanumeric() || c == b'_')
        && !KEYWORDS
            .split_ascii_whitespace()
            .any(|keyword| keyword == name)
}
fn generated(name: &str) -> bool {
    matches!(name, "clk" | "rst_n")
        || name.starts_with("sc_")
        || name.starts_with("encode_")
        || name.starts_with("decode_")
        || name.starts_with("dense_")
        || name.starts_with("u_")
        || ((name.starts_with('v') || name.starts_with('c'))
            && name.as_bytes().get(1).is_some_and(u8::is_ascii_digit))
}
pub(crate) fn verify(graph: &ScGraph, errors: &mut Vec<VerifyError>) {
    if graph.name == "SYNTHESIS" {
        errors.push(VerifyError {
            op_index: 0,
            message: "HDL module identifier is reserved by CIRCT: SYNTHESIS".into(),
        });
    }
    if !valid(&graph.name) {
        errors.push(VerifyError {
            op_index: 0,
            message: format!("invalid HDL module identifier: {:?}", graph.name),
        });
    }
    let mut ports = HashSet::new();
    for (op_index, op) in graph.ops.iter().enumerate() {
        if let ScOp::Input { name, .. } | ScOp::Output { name, .. } = op {
            let reason = if name == "SYNTHESIS" {
                Some("HDL port identifier is reserved by CIRCT")
            } else if !valid(name) {
                Some("invalid HDL port identifier")
            } else if generated(name) {
                Some("HDL port identifier uses a reserved generated namespace")
            } else if !ports.insert(name) {
                Some("duplicate HDL port identifier")
            } else {
                None
            };
            if let Some(reason) = reason {
                errors.push(VerifyError {
                    op_index,
                    message: format!("{reason}: {name:?}"),
                });
            }
        }
    }
}
