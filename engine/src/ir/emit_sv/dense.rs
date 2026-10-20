// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense row-major matrix lowering

//! Lower DenseForward to the maintained row-major matrix core.

use super::{
    emit_dense_fold_plan_comment, emit_target_dsp_attribute, probability, signed_q_literal,
    value_to_wire, DenseParams, ScGraph, SvTarget, ValueId,
};

/// Instantiate the matrix core with separate probability and state word geometry.
pub(super) fn emit(
    sv: &mut String,
    graph: &ScGraph,
    inst: u32,
    id: ValueId,
    operands: [ValueId; 4],
    p: &DenseParams,
    target: &SvTarget,
) -> Result<(), String> {
    let [inputs, weights, leak, gain] = operands;
    let input_bus = format!("dense_{inst}_inputs");
    let weight_bus = format!("dense_{inst}_weights");
    probability::emit(sv, graph, inputs, p.n_inputs, &input_bus)?;
    probability::emit(sv, graph, weights, p.n_inputs * p.n_neurons, &weight_bus)?;
    emit_dense_fold_plan_comment(sv, target, p);
    emit_target_dsp_attribute(sv, target);
    sv.push_str(&format!(
        "    sc_dense_matrix_layer #(\n\
         \x20       .N_INPUTS({ni}),\n\
         \x20       .N_NEURONS({nn}),\n\
         \x20       .DATA_WIDTH({dw}),\n\
         \x20       .FRACTION(8),\n\
         \x20       .PROBABILITY_WIDTH(16),\n\
         \x20       .INPUT_SEED_BASE(16'h{input_seed:04X}),\n\
         \x20       .WEIGHT_SEED_BASE(16'h{weight_seed:04X})\n\
         \x20   ) u_dense_{inst} (\n\
         \x20       .clk(clk), .rst_n(rst_n),\n\
         \x20       .start_pulse(1'b1), .stream_len(32'd{length}),\n\
         \x20       .x_input_fp({input_bus}), .weight_fp({weight_bus}),\n\
         \x20       .y_min_fp({ymin}), .y_max_fp({ymax}),\n\
         \x20       .cfg_leak({leak}), .cfg_gain({gain}),\n\
         \x20       .spikes(v{result}_spikes), .step_valid(),\n\
         \x20       .run_done(v{result}_done), .running(v{result}_running)\n\
         \x20   );\n\n",
        ni = p.n_inputs,
        nn = p.n_neurons,
        dw = p.data_width,
        input_seed = p.input_seed_base,
        weight_seed = p.weight_seed_base,
        length = p.stream_length,
        ymin = signed_q_literal(p.y_min, p.data_width as usize),
        ymax = signed_q_literal(p.y_max, p.data_width as usize),
        leak = value_to_wire(graph, leak),
        gain = value_to_wire(graph, gain),
        result = id.0,
    ));
    Ok(())
}
