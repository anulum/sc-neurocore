// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for verilog_generator

//! Validated top-level Verilog and clocked-AER emitter mirror for `sc_dense_layer_core`.

#![allow(non_snake_case)]

const RESERVED_IDENTIFIERS: [&str; 11] = [
    "always",
    "assign",
    "begin",
    "case",
    "endmodule",
    "input",
    "module",
    "output",
    "reg",
    "wire",
    "xor",
];

/// Configuration for one `sc_dense_layer_core` stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLayerParams {
    /// Number of postsynaptic LIF neurons (`N_NEURONS`).
    pub n_neurons: usize,
    /// Optional explicit incoming bitstream width (`N_INPUTS`).
    pub input_width: Option<usize>,
    /// Optional explicit outgoing spike bus width (must equal `n_neurons`).
    pub output_width: Option<usize>,
}

impl DenseLayerParams {
    /// Construct dense-layer parameters with default inferred bus widths.
    pub fn new(n_neurons: usize) -> Self {
        Self {
            n_neurons,
            input_width: None,
            output_width: None,
        }
    }
}

/// Supported standalone 16-bit stochastic bitstream source generators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticSourceKind {
    /// Maximal-length 16-bit linear-feedback shift register.
    Lfsr16,
    /// 16-bit Gray-code Sobol quasi-random generator.
    Sobol16,
    /// 16-bit bit-reversed base-2 Halton quasi-random generator.
    Halton16,
}

/// Declared layer entry inside [`VerilogGenerator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerDefinition {
    /// Clocked stochastic dense layer backed by `sc_dense_layer_core`.
    Dense {
        /// Sanitised Verilog instance prefix.
        name: String,
        /// Validated neuron and bus-width parameters.
        params: DenseLayerParams,
    },
    /// Standalone stochastic bitstream source module.
    StochasticSource {
        /// Sanitised Verilog module identifier.
        name: String,
        /// Generator algorithm family.
        kind: StochasticSourceKind,
        /// Initial 16-bit generator seed.
        seed: u16,
    },
    /// Placeholder entry retained so unsupported layer types fail closed at emission.
    Unsupported {
        /// Raw layer-type label.
        layer_type: String,
        /// Sanitised layer identifier.
        name: String,
    },
}

/// Top-level Verilog emitter mirroring `sc_neurocore.hdl_gen.VerilogGenerator`.
#[derive(Debug, Clone)]
pub struct VerilogGenerator {
    /// Sanitised top-level Verilog module identifier.
    pub module_name: String,
    /// Default input/output bitstream width when no layers override it.
    pub bus_width: usize,
    /// Ordered layer definitions.
    pub layers: Vec<LayerDefinition>,
}

impl Default for VerilogGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl VerilogGenerator {
    /// Construct a generator with module name `"sc_network_top"` and 8-bit bus width.
    pub fn new() -> Self {
        Self::try_new("sc_network_top", 8).expect("default Verilog generator parameters are valid")
    }

    /// Construct a generator after validating `module_name` and `bus_width`.
    pub fn try_new(module_name: &str, bus_width: usize) -> Result<Self, String> {
        Ok(Self {
            module_name: sanitize_ident(module_name, "module name")?,
            bus_width: require_positive_width(bus_width, "bus_width")?,
            layers: Vec::new(),
        })
    }

    /// Validate and append a `Dense` layer specification.
    pub fn add_dense_layer(&mut self, name: &str, params: DenseLayerParams) -> Result<(), String> {
        self.add_layer(LayerDefinition::Dense {
            name: sanitize_ident(name, "layer name")?,
            params: validate_dense_params(params, name)?,
        })
    }

    /// Validate and append a standalone stochastic source specification.
    pub fn add_stochastic_source(
        &mut self,
        name: &str,
        kind: StochasticSourceKind,
        seed: u16,
    ) -> Result<(), String> {
        self.add_layer(LayerDefinition::StochasticSource {
            name: sanitize_ident(name, "stochastic source module name")?,
            kind,
            seed,
        })
    }

    /// Validate identifiers and append an arbitrary [`LayerDefinition`].
    pub fn add_layer(&mut self, layer: LayerDefinition) -> Result<(), String> {
        match &layer {
            LayerDefinition::Dense { name, params } => {
                sanitize_ident(name, "layer name")?;
                validate_dense_params(*params, name)?;
            }
            LayerDefinition::StochasticSource { name, .. } => {
                sanitize_ident(name, "stochastic source module name")?;
            }
            LayerDefinition::Unsupported { name, .. } => {
                sanitize_ident(name, "layer name")?;
            }
        }
        self.layers.push(layer);
        Ok(())
    }

    /// Emit complete Verilog source for `"sync"` or `"async_aer"` mode.
    pub fn generate(&self, mode: &str) -> Result<String, String> {
        if mode == "async_aer" {
            return self.generate_async_aer();
        }
        if mode != "sync" {
            return Err("mode must be 'sync' or 'async_aer'".to_string());
        }
        self.validate_sync_layers()?;
        let widths = self.sync_layer_widths()?;
        let input_width = widths.first().map_or(self.bus_width, |width| width.0);
        let output_width = widths.last().map_or(self.bus_width, |width| width.1);

        let mut code = String::new();
        code.push_str(&format!("module {} (\n", self.module_name));
        code.push_str("    input wire clk,\n");
        code.push_str("    input wire rst_n,\n");
        code.push_str(&format!(
            "    input wire [{}:0] input_bus,\n",
            input_width - 1
        ));
        code.push_str(&format!(
            "    output wire [{}:0] output_bus\n",
            output_width - 1
        ));
        code.push_str(");\n\n");

        code.push_str("    // Internal Signals\n");
        for (i, width) in widths
            .iter()
            .enumerate()
            .take(widths.len().saturating_sub(1))
        {
            code.push_str(&format!(
                "    wire [{}:0] layer_{i}_to_{};\n",
                width.1 - 1,
                i + 1
            ));
        }
        if !widths.is_empty() {
            code.push_str("    genvar sc_in_idx;\n");
        }
        code.push('\n');

        let mut dense_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let LayerDefinition::Dense { name, params } = layer {
                let in_bus = if dense_idx == 0 {
                    "input_bus".to_string()
                } else {
                    format!("layer_{}_to_{dense_idx}", dense_idx - 1)
                };
                let out_bus = if dense_idx == widths.len() - 1 {
                    "output_bus".to_string()
                } else {
                    format!("layer_{dense_idx}_to_{}", dense_idx + 1)
                };
                code.push_str(&render_dense_layer_instance(
                    &format!("Layer {layer_idx}: {name}"),
                    name,
                    params.n_neurons,
                    widths[dense_idx].0,
                    &in_bus,
                    &out_bus,
                ));
                code.push('\n');
                dense_idx += 1;
            }
        }

        if widths.is_empty() {
            code.push_str(&format!("    assign output_bus = {output_width}'b0;\n"));
            code.push_str(
                "    wire _unused_empty_ports = &{1'b0, clk, rst_n, input_bus, 1'b0};\n\n",
            );
        }

        code.push_str("endmodule\n");
        let source_modules = self.emit_sources()?;
        if !source_modules.is_empty() {
            code.push_str("\n\n");
            code.push_str(&source_modules);
            code.push('\n');
        }
        Ok(code)
    }

    fn generate_async_aer(&self) -> Result<String, String> {
        self.validate_async_aer_layers()?;
        let widths = self.sync_layer_widths()?;
        let input_width = widths.first().map_or(self.bus_width, |width| width.0);
        let spike_width = widths.last().map_or(self.bus_width, |width| width.1);
        let addr_width = ceil_log2(spike_width).max(1);

        let mut code = String::new();
        code.push_str(&format!("module {} (\n", self.module_name));
        code.push_str("    input wire clk,\n");
        code.push_str("    input wire rst_n,\n");
        code.push_str(&format!(
            "    input wire [{}:0] input_bus,\n",
            input_width - 1
        ));
        code.push_str("    input wire aer_ack,\n");
        code.push_str("    output reg aer_req,\n");
        code.push_str(&format!(
            "    output reg [{}:0] aer_addr,\n",
            addr_width - 1
        ));
        code.push_str(&format!(
            "    output wire [{}:0] output_bus\n",
            spike_width - 1
        ));
        code.push_str(");\n\n");

        code.push_str("    // Research boundary: sync compute path with AER output wrapper.\n");
        code.push_str("    // This is not a full asynchronous micropipeline implementation.\n");
        code.push_str(&format!("    wire [{}:0] spike_vector;\n", spike_width - 1));
        for (i, width) in widths
            .iter()
            .enumerate()
            .take(widths.len().saturating_sub(1))
        {
            code.push_str(&format!(
                "    wire [{}:0] layer_{i}_to_{};\n",
                width.1 - 1,
                i + 1
            ));
        }
        if !widths.is_empty() {
            code.push_str("    genvar sc_in_idx;\n");
        }
        code.push('\n');

        let mut dense_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let LayerDefinition::Dense { name, params } = layer else {
                continue;
            };
            let input_bus = if dense_idx == 0 {
                "input_bus".to_string()
            } else {
                format!("layer_{}_to_{dense_idx}", dense_idx - 1)
            };
            let output_bus = if dense_idx == widths.len() - 1 {
                "spike_vector".to_string()
            } else {
                format!("layer_{dense_idx}_to_{}", dense_idx + 1)
            };
            code.push_str(&render_dense_layer_instance(
                &format!("Sync layer {layer_idx}: {name}"),
                name,
                params.n_neurons,
                widths[dense_idx].0,
                &input_bus,
                &output_bus,
            ));
            code.push('\n');
            dense_idx += 1;
        }

        if widths.is_empty() {
            code.push_str(&format!("    assign spike_vector = {spike_width}'b0;\n"));
            code.push_str("    wire _unused_empty_inputs = &{1'b0, input_bus, 1'b0};\n\n");
        }

        code.push_str("    assign output_bus = spike_vector;\n");
        code.push_str("    wire spike_valid = |spike_vector;\n\n");
        code.push_str(&format!(
            "    function [{}:0] first_hot_index;\n",
            addr_width - 1
        ));
        code.push_str(&format!("        input [{}:0] vector;\n", spike_width - 1));
        code.push_str("        integer k;\n");
        code.push_str("        reg found;\n");
        code.push_str("        begin\n");
        code.push_str(&format!("            first_hot_index = {addr_width}'d0;\n"));
        code.push_str("            found = 1'b0;\n");
        code.push_str(&format!(
            "            for (k = 0; k < {spike_width}; k = k + 1) begin\n"
        ));
        code.push_str("                if (!found && vector[k]) begin\n");
        code.push_str(&format!(
            "                    first_hot_index = k[{}:0];\n",
            addr_width - 1
        ));
        code.push_str("                    found = 1'b1;\n");
        code.push_str("                end\n");
        code.push_str("            end\n");
        code.push_str("        end\n");
        code.push_str("    endfunction\n\n");

        code.push_str(&format!(
            "    wire [{}:0] encoded_addr = first_hot_index(spike_vector);\n",
            addr_width - 1
        ));
        code.push_str(&format!("    reg [{}:0] event_vector;\n", spike_width - 1));
        code.push_str(&format!(
            "    reg [{}:0] acknowledged_vector;\n",
            spike_width - 1
        ));
        code.push_str(
            "    wire new_spike_vector = spike_valid && (spike_vector != acknowledged_vector);\n\n",
        );
        code.push_str("    always @(posedge clk or negedge rst_n) begin\n");
        code.push_str("        if (!rst_n) begin\n");
        code.push_str("            aer_req <= 1'b0;\n");
        code.push_str(&format!("            aer_addr <= {addr_width}'d0;\n"));
        code.push_str(&format!("            event_vector <= {spike_width}'d0;\n"));
        code.push_str(&format!(
            "            acknowledged_vector <= {spike_width}'d0;\n"
        ));
        code.push_str("        end else begin\n");
        code.push_str("            if (!spike_valid) begin\n");
        code.push_str(&format!(
            "                acknowledged_vector <= {spike_width}'d0;\n"
        ));
        code.push_str("            end\n");
        code.push_str("            if (!aer_req && new_spike_vector) begin\n");
        code.push_str("                aer_req <= 1'b1;\n");
        code.push_str("                aer_addr <= encoded_addr;\n");
        code.push_str("                event_vector <= spike_vector;\n");
        code.push_str("            end else if (aer_req && aer_ack) begin\n");
        code.push_str("                aer_req <= 1'b0;\n");
        code.push_str("                acknowledged_vector <= event_vector;\n");
        code.push_str("            end\n");
        code.push_str("        end\n");
        code.push_str("    end\n\n");
        code.push_str("endmodule\n");
        Ok(code)
    }

    /// Persist synchronous generated Verilog to `path`.
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.generate("sync")?)
            .map_err(|err| format!("failed to write Verilog to {path}: {err}"))
    }

    /// Emit a standalone LFSR-16 stochastic source module.
    pub fn emit_lfsr16_source(&self, module_name: &str, seed: u16) -> Result<String, String> {
        emit_lfsr16_source(module_name, seed)
    }

    /// Emit a standalone Sobol-16 stochastic source module.
    pub fn emit_sobol16_source(&self, module_name: &str, seed: u16) -> Result<String, String> {
        emit_sobol16_source(module_name, seed)
    }

    /// Emit a standalone Halton-16 stochastic source module.
    pub fn emit_halton16_source(&self, module_name: &str) -> Result<String, String> {
        emit_halton16_source(module_name)
    }

    fn validate_sync_layers(&self) -> Result<(), String> {
        for layer in &self.layers {
            match layer {
                LayerDefinition::Dense { name, params } => {
                    validate_dense_params(*params, name)?;
                }
                LayerDefinition::StochasticSource { .. } => {}
                LayerDefinition::Unsupported { layer_type, name } => {
                    return Err(format!(
                        "unsupported sync layer type '{layer_type}' for layer '{name}'"
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_async_aer_layers(&self) -> Result<(), String> {
        for layer in &self.layers {
            match layer {
                LayerDefinition::Dense { name, params } => {
                    validate_dense_params(*params, name)?;
                }
                LayerDefinition::StochasticSource { name, .. } => {
                    return Err(format!(
                        "unsupported async AER layer type 'StochasticSource' for layer '{name}'"
                    ));
                }
                LayerDefinition::Unsupported { layer_type, name } => {
                    return Err(format!(
                        "unsupported async AER layer type '{layer_type}' for layer '{name}'"
                    ));
                }
            }
        }
        Ok(())
    }

    fn sync_layer_widths(&self) -> Result<Vec<(usize, usize)>, String> {
        let mut widths = Vec::new();
        let mut previous_width: Option<usize> = None;
        let mut previous_name: Option<&str> = None;
        let mut seen_dense = Vec::<&str>::new();

        for layer in &self.layers {
            let LayerDefinition::Dense { name, params } = layer else {
                continue;
            };
            if seen_dense.contains(&name.as_str()) {
                return Err(format!("duplicate layer name {name:?}"));
            }
            seen_dense.push(name);
            let input_width = params
                .input_width
                .unwrap_or(previous_width.unwrap_or(self.bus_width));
            let output_width = params.output_width.unwrap_or(params.n_neurons);
            require_positive_width(input_width, "input_width")?;
            require_positive_width(output_width, "output_width")?;

            if let Some(previous) = previous_width {
                if input_width != previous {
                    return Err(format!(
                        "{} -> {name} width mismatch: {previous} output bits cannot drive {input_width} input bits",
                        previous_name.unwrap_or("<previous>")
                    ));
                }
            }
            widths.push((input_width, output_width));
            previous_width = Some(output_width);
            previous_name = Some(name);
        }
        Ok(widths)
    }

    fn emit_sources(&self) -> Result<String, String> {
        let mut seen = Vec::<&str>::new();
        let mut emitted = Vec::new();
        for layer in &self.layers {
            let LayerDefinition::StochasticSource { name, kind, seed } = layer else {
                continue;
            };
            if seen.contains(&name.as_str()) {
                return Err(format!("duplicate stochastic source module name {name:?}"));
            }
            seen.push(name);
            emitted.push(match kind {
                StochasticSourceKind::Lfsr16 => emit_lfsr16_source(name, *seed)?,
                StochasticSourceKind::Sobol16 => emit_sobol16_source(name, *seed)?,
                StochasticSourceKind::Halton16 => emit_halton16_source(name)?,
            });
        }
        Ok(emitted.join("\n\n"))
    }
}

fn render_dense_layer_instance(
    comment_label: &str,
    layer_name: &str,
    n_neurons: usize,
    input_width: usize,
    input_bus: &str,
    output_bus: &str,
) -> String {
    let dw = 16usize;
    let lines = [
        format!("    // {comment_label}"),
        format!("    wire [{}:0] {layer_name}_x_input_fp;", input_width * dw - 1),
        format!("    wire [{}:0] {layer_name}_i_t;", dw - 1),
        format!("    wire {layer_name}_step_valid;"),
        format!("    wire {layer_name}_run_done;"),
        format!("    wire {layer_name}_running;"),
        "    generate".to_string(),
        format!(
            "        for (sc_in_idx = 0; sc_in_idx < {input_width}; sc_in_idx = sc_in_idx + 1) begin : {layer_name}_pack_x"
        ),
        format!(
            "            assign {layer_name}_x_input_fp[sc_in_idx*{dw} +: {dw}] = {{{dw}{{{input_bus}[sc_in_idx]}}}};"
        ),
        "        end".to_string(),
        "    endgenerate".to_string(),
        "    sc_dense_layer_core #(".to_string(),
        format!("        .N_INPUTS({input_width}),"),
        format!("        .N_NEURONS({n_neurons}),"),
        format!("        .DATA_WIDTH({dw})"),
        format!("    ) {layer_name}_inst ("),
        "        .clk(clk),".to_string(),
        "        .rst_n(rst_n),".to_string(),
        "        .start_pulse(1'b1),".to_string(),
        "        .stream_len(32'd256),".to_string(),
        format!("        .x_input_fp({layer_name}_x_input_fp),"),
        format!("        .weight_fp({{{input_width}{{16'hFFFF}}}}),"),
        "        .y_min_fp(16'h0000),".to_string(),
        "        .y_max_fp(16'h0200),".to_string(),
        "        .cfg_leak(16'h0010),".to_string(),
        "        .cfg_gain(16'h0100),".to_string(),
        format!("        .I_t({layer_name}_i_t),"),
        format!("        .spikes({output_bus}),"),
        format!("        .step_valid({layer_name}_step_valid),"),
        format!("        .run_done({layer_name}_run_done),"),
        format!("        .running({layer_name}_running)"),
        "    );".to_string(),
        format!(
            "    wire _unused_{layer_name}_status = &{{1'b0, {layer_name}_i_t, {layer_name}_step_valid, {layer_name}_run_done, {layer_name}_running, 1'b0}};"
        ),
        String::new(),
    ];
    lines.join("\n")
}

fn validate_dense_params(params: DenseLayerParams, name: &str) -> Result<DenseLayerParams, String> {
    require_positive_width(params.n_neurons, &format!("Dense layer '{name}' n_neurons"))?;
    if let Some(width) = params.input_width {
        require_positive_width(width, &format!("Dense layer '{name}' input_width"))?;
    }
    if let Some(width) = params.output_width {
        require_positive_width(width, &format!("Dense layer '{name}' output_width"))?;
        if width != params.n_neurons {
            return Err(format!(
                "Dense layer '{name}' output_width ({width}) must equal n_neurons ({})",
                params.n_neurons
            ));
        }
    }
    Ok(params)
}

fn require_positive_width(value: usize, name: &str) -> Result<usize, String> {
    if value == 0 {
        Err(format!("{name} must be a positive integer"))
    } else {
        Ok(value)
    }
}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

fn sanitize_ident(value: &str, context: &str) -> Result<String, String> {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.is_empty() || chars.len() > 64 {
        return Err(format!("Invalid identifier for {context}: {value:?}"));
    }
    let first = chars[0];
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(format!("Invalid identifier for {context}: {value:?}"));
    }
    if !chars
        .iter()
        .all(|ch| *ch == '_' || ch.is_ascii_alphanumeric())
    {
        return Err(format!("Invalid identifier for {context}: {value:?}"));
    }
    if RESERVED_IDENTIFIERS.contains(&value) {
        return Err(format!("Invalid identifier for {context}: {value:?}"));
    }
    Ok(value.to_string())
}

fn emit_lfsr16_source(module_name: &str, seed: u16) -> Result<String, String> {
    let module_name = sanitize_ident(module_name, "module name")?;
    let seed = if seed == 0 { 0xACE1 } else { seed };
    let first_sample = lfsr16_advance(seed);
    let lines = [
        format!("module {module_name} ("),
        "    input wire clk,".to_string(),
        "    input wire rst_n,".to_string(),
        "    input wire [15:0] threshold,".to_string(),
        "    output wire bit_out,".to_string(),
        "    output reg [15:0] state".to_string(),
        ");".to_string(),
        String::new(),
        format!("    localparam [15:0] SEED = 16'h{seed:04X};"),
        format!("    localparam [15:0] FIRST_SAMPLE = 16'h{first_sample:04X};"),
        "    wire feedback;".to_string(),
        String::new(),
        "    // Compare the current generated sample before the next advance.".to_string(),
        "    assign bit_out = (state < threshold);".to_string(),
        "    assign feedback = state[0] ^ state[2] ^ state[3] ^ state[5];".to_string(),
        String::new(),
        "    always @(posedge clk or negedge rst_n) begin".to_string(),
        "        if (!rst_n) begin".to_string(),
        "            state <= FIRST_SAMPLE;".to_string(),
        "        end else begin".to_string(),
        "            state <= {feedback, state[15:1]};".to_string(),
        "        end".to_string(),
        "    end".to_string(),
        "endmodule".to_string(),
    ];
    Ok(lines.join("\n"))
}

fn lfsr16_advance(state: u16) -> u16 {
    let feedback = (state ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1;
    (state >> 1) | (feedback << 15)
}

fn emit_sobol16_source(module_name: &str, seed: u16) -> Result<String, String> {
    let module_name = sanitize_ident(module_name, "module name")?;
    let first_sample = seed ^ 0x8000;
    let direction_cases = [
        ("16'b???????????????1", "16'h8000"),
        ("16'b??????????????10", "16'h4000"),
        ("16'b?????????????100", "16'h2000"),
        ("16'b????????????1000", "16'h1000"),
        ("16'b???????????10000", "16'h0800"),
        ("16'b??????????100000", "16'h0400"),
        ("16'b?????????1000000", "16'h0200"),
        ("16'b????????10000000", "16'h0100"),
        ("16'b???????100000000", "16'h0080"),
        ("16'b??????1000000000", "16'h0040"),
        ("16'b?????10000000000", "16'h0020"),
        ("16'b????100000000000", "16'h0010"),
        ("16'b???1000000000000", "16'h0008"),
        ("16'b??10000000000000", "16'h0004"),
        ("16'b?100000000000000", "16'h0002"),
        ("16'b1000000000000000", "16'h0001"),
    ];
    let mut lines = vec![
        format!("module {module_name} ("),
        "    input wire clk,".to_string(),
        "    input wire rst_n,".to_string(),
        "    input wire [15:0] threshold,".to_string(),
        "    output wire bit_out,".to_string(),
        "    output reg [15:0] value,".to_string(),
        "    output reg [15:0] index".to_string(),
        ");".to_string(),
        String::new(),
        format!("    localparam [15:0] SEED = 16'h{seed:04X};"),
        format!("    localparam [15:0] FIRST_SAMPLE = 16'h{first_sample:04X};"),
        String::new(),
        "    reg [15:0] direction;".to_string(),
        String::new(),
        "    // Compare the current generated sample before the next advance.".to_string(),
        "    assign bit_out = (value < threshold);".to_string(),
        String::new(),
        "    always @(*) begin".to_string(),
        "        casez (index)".to_string(),
    ];
    for (pattern, direction) in direction_cases {
        lines.push(format!("            {pattern}: direction = {direction};"));
    }
    lines.extend([
        "            default: direction = 16'h8000;".to_string(),
        "        endcase".to_string(),
        "    end".to_string(),
        String::new(),
        "    always @(posedge clk or negedge rst_n) begin".to_string(),
        "        if (!rst_n) begin".to_string(),
        "            value <= FIRST_SAMPLE;".to_string(),
        "            index <= 16'd1;".to_string(),
        "        end else begin".to_string(),
        "            value <= value ^ direction;".to_string(),
        "            index <= index + 16'd1;".to_string(),
        "        end".to_string(),
        "    end".to_string(),
        "endmodule".to_string(),
    ]);
    Ok(lines.join("\n"))
}

fn emit_halton16_source(module_name: &str) -> Result<String, String> {
    let module_name = sanitize_ident(module_name, "module name")?;
    let mut lines = vec![
        format!("module {module_name} ("),
        "    input wire clk,".to_string(),
        "    input wire rst_n,".to_string(),
        "    input wire enable,".to_string(),
        "    output reg [15:0] quasi_random,".to_string(),
        "    output reg valid".to_string(),
        ");".to_string(),
        String::new(),
        "    reg [15:0] counter;".to_string(),
        String::new(),
        "    // Bit-reversal = Van der Corput base-2 radical inverse".to_string(),
        "    // Pure routing — zero LUT cost".to_string(),
        "    wire [15:0] reversed;".to_string(),
        String::new(),
    ];
    for idx in 0..16 {
        lines.push(format!(
            "    assign reversed[{idx}] = counter[{}];",
            15 - idx
        ));
    }
    lines.extend([
        String::new(),
        "    always @(posedge clk or negedge rst_n) begin".to_string(),
        "        if (!rst_n) begin".to_string(),
        "            counter      <= 16'd0;".to_string(),
        "            quasi_random <= 16'd0;".to_string(),
        "            valid        <= 1'b0;".to_string(),
        "        end else if (enable) begin".to_string(),
        "            quasi_random <= reversed;".to_string(),
        "            valid        <= 1'b1;".to_string(),
        "            counter      <= counter + 16'd1;".to_string(),
        "        end else begin".to_string(),
        "            valid <= 1'b0;".to_string(),
        "        end".to_string(),
        "    end".to_string(),
        "endmodule".to_string(),
    ]);
    Ok(lines.join("\n"))
}

/// Return `true` when `state` holds valid identifiers, bus widths, and layer parameters.
pub fn validate_verilog_generator(state: &VerilogGenerator) -> bool {
    !state.module_name.is_empty()
        && state.bus_width > 0
        && sanitize_ident(&state.module_name, "module name").is_ok()
        && state.layers.iter().all(|layer| match layer {
            LayerDefinition::Dense { name, params } => {
                sanitize_ident(name, "layer name").is_ok()
                    && validate_dense_params(*params, name).is_ok()
            }
            LayerDefinition::StochasticSource { name, .. } => {
                sanitize_ident(name, "stochastic source module name").is_ok()
            }
            LayerDefinition::Unsupported { name, .. } => sanitize_ident(name, "layer name").is_ok(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verilog_generator_new() {
        let state = VerilogGenerator::new();
        assert!(validate_verilog_generator(&state));
        assert_eq!(state.module_name, "sc_network_top");
        assert_eq!(state.bus_width, 8);
    }

    #[test]
    fn test_verilog_generator_derives_declared_bus_widths() {
        let mut generator = VerilogGenerator::try_new("wide_top", 16).unwrap();
        generator
            .add_dense_layer(
                "dense0",
                DenseLayerParams {
                    n_neurons: 12,
                    input_width: None,
                    output_width: Some(12),
                },
            )
            .unwrap();

        let verilog = generator.generate("sync").unwrap();

        assert!(verilog.contains("module wide_top"));
        assert!(verilog.contains("input wire [15:0] input_bus"));
        assert!(verilog.contains("output wire [11:0] output_bus"));
        assert!(verilog.contains(".N_INPUTS(16)"));
        assert!(verilog.contains(".N_NEURONS(12)"));
        assert!(verilog.contains(".spikes(output_bus)"));
    }

    #[test]
    fn test_verilog_generator_rejects_mismatched_dense_widths() {
        let mut generator = VerilogGenerator::new();
        generator
            .add_dense_layer(
                "dense0",
                DenseLayerParams {
                    n_neurons: 5,
                    input_width: None,
                    output_width: Some(5),
                },
            )
            .unwrap();
        generator
            .add_dense_layer(
                "dense1",
                DenseLayerParams {
                    n_neurons: 3,
                    input_width: Some(7),
                    output_width: None,
                },
            )
            .unwrap();

        let err = generator.generate("sync").unwrap_err();
        assert!(err.contains("dense0 -> dense1 width mismatch"));
    }

    #[test]
    fn test_verilog_generator_rejects_invalid_identifiers_and_params() {
        assert!(VerilogGenerator::try_new("test); malicious_module(", 8).is_err());
        assert!(VerilogGenerator::try_new("safe_top", 0).is_err());

        let mut generator = VerilogGenerator::new();
        assert!(generator
            .add_dense_layer("bad-name", DenseLayerParams::new(4))
            .is_err());
        assert!(generator
            .add_dense_layer("dense0", DenseLayerParams::new(0))
            .is_err());
        assert!(generator
            .add_dense_layer(
                "dense_mismatch",
                DenseLayerParams {
                    n_neurons: 4,
                    input_width: None,
                    output_width: Some(2),
                },
            )
            .is_err());
    }

    #[test]
    fn test_verilog_generator_routes_stochastic_source_layers() {
        let mut generator = VerilogGenerator::new();
        generator
            .add_stochastic_source("rng_lfsr", StochasticSourceKind::Lfsr16, 0xBEEF)
            .unwrap();
        generator
            .add_stochastic_source("rng_sobol", StochasticSourceKind::Sobol16, 0x0042)
            .unwrap();

        let verilog = generator.generate("sync").unwrap();

        assert!(verilog.contains("module rng_lfsr"));
        assert!(verilog.contains("16'hBEEF"));
        assert!(verilog.contains("module rng_sobol"));
        assert!(verilog.contains("16'h0042"));
    }

    #[test]
    fn test_verilog_generator_async_aer_wraps_dense_path() {
        let mut generator = VerilogGenerator::try_new("async_wrap", 8).unwrap();
        generator
            .add_dense_layer("dense0", DenseLayerParams::new(4))
            .unwrap();

        let verilog = generator.generate("async_aer").unwrap();

        assert!(verilog.contains("module async_wrap"));
        assert!(verilog.contains("input wire aer_ack"));
        assert!(verilog.contains("output reg aer_req"));
        assert!(verilog.contains("output reg [1:0] aer_addr"));
        assert!(verilog.contains(".spikes(spike_vector)"));
    }

    #[test]
    fn test_verilog_generator_emits_halton16_source() {
        let generator = VerilogGenerator::new();
        let verilog = generator.emit_halton16_source("rng_halton").unwrap();
        assert!(verilog.contains("module rng_halton"));
        assert!(verilog.contains("reversed"));
    }

    #[test]
    fn test_verilog_generator_rejects_unsupported_modes_and_layers() {
        let mut generator = VerilogGenerator::new();
        generator
            .add_layer(LayerDefinition::Unsupported {
                layer_type: "Custom".to_string(),
                name: "custom0".to_string(),
            })
            .unwrap();

        assert!(generator.generate("async_aer").is_err());
        let err = generator.generate("sync").unwrap_err();
        assert!(err.contains("unsupported sync layer type 'Custom'"));
    }
}
