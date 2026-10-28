// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Actual native compiler result fixtures for frontend transitions

import type { IRBuildResponse, IRVerifyResponse, SVEmitResponse } from "./api/equationIrTypes";

// Captured from the production ASGI routes and real supervised native workers.
// Source: dv/dt = I, initial v=0, dt=0.5, signed total width16/fraction8.
const built: IRBuildResponse = {
  "operation": "build",
  "ir_text": "sc.graph @sc_equation_neuron {\n  sc.source v1 source=2a1dba0643f5ff11a5911c5587f2fe4608ccc53837c8f41559f31c837ecb613f graph=afb10d1ff452f382940ee1c68e94a00c8b6e8c9a84f8aa630439c008e9550781 payload=7b226d6f64656c223a7b22626173655f73746174655f756e697473223a7b7d2c22636f6e7374616e7473223a5b5d2c22646574656374696f6e223a226c6576656c222c22646973706c61795f73746174655f756e697473223a7b7d2c226474223a302e352c226571756174696f6e73223a5b5b2276222c2249225d5d2c22696e697469616c5f7374617465223a5b5b2276222c305d5d2c226d6574686f64223a2265756c6572222c22706172616d6574657273223a5b5d2c2270726f626162696c6974795f65787072657373696f6e223a6e756c6c2c2272616e646f6d5f73747265616d223a302c22726174655f65787072657373696f6e223a6e756c6c2c227265736574223a5b5d2c22726e675f73656564223a34343235372c2272756e74696d655f756e697473223a7b7d2c227370696b655f76616c7565223a6e756c6c2c227375627374657073223a312c227468726573686f6c64223a6e756c6c2c227468726573686f6c645f726e67223a226c6673723136222c22756e697473223a226e6f6e65227d2c226d6f64756c655f6e616d65223a2273635f6571756174696f6e5f6e6575726f6e222c226e756d65726963223a7b22646174615f7769647468223a31362c226672616374696f6e223a382c2266756e6374696f6e5f6c6f776572696e67223a226c75742d7631222c226f766572666c6f77223a227361747572617465222c2272616e646f6d5f707265636973696f6e223a227374616e64617264222c22726f756e64696e67223a227472756e63617465222c227369676e6564223a747275657d2c22706970656c696e65223a7b22706f696e7473223a5b5d2c22737461676573223a307d2c22736368656d61223a2273632d6571756174696f6e2d736f757263652e7631222c2273656d616e74696373223a7b226469766973696f6e223a22656e636f6465642d6e6f6e7a65726f2d6f722d736f757263652d6775617264222c226576656e745f6f766572666c6f77223a227361747572617465222c227265736574223a226173796e6368726f6e6f75732d6163746976652d6c6f772d696e697469616c2d7374617465222c22726e67223a226d6f64656c2d7365656436342d73747265616d33322d736974652d636f756e746572222c2273746174655f6f766572666c6f77223a227361747572617465227d7d\n  %0 = sc.constant 0 : i16\n  %1 = sc.state %0 : i16\n  %2 = sc.input \"I_t\" : i16\n  %3 = sc.word resize %2 : i32\n  %4 = sc.word resize %3 : i32\n  %5 = sc.constant 128 : i32\n  %6 = sc.word mul %4, %5 : i32\n  %7 = sc.constant 8 : i32\n  %8 = sc.word ashr %6, %7 : i32\n  %9 = sc.word resize %8 : i16\n  %10 = sc.word resize %1 : i17\n  %11 = sc.word resize %9 : i17\n  %12 = sc.word add %10, %11 : i17\n  %13 = sc.constant -32768 : i17\n  %14 = sc.constant 32767 : i17\n  %15 = sc.word cmp_ge %12, %13 : bool\n  %16 = sc.word cmp_le %12, %14 : bool\n  %17 = sc.word select %16, %12, %14 : i17\n  %18 = sc.word select %15, %17, %13 : i17\n  %19 = sc.word resize %18 : i16\n  %20 = sc.constant 0 : i1\n  %21 = sc.word cmp_eq %20, %20 : bool\n  %22 = sc.constant i64(0) : bool\n  %23 = sc.word select %21, %22, %22 : bool\n  %24 = sc.store %1, %19, enable=%21\n  %25 = sc.constant i64(0) : bool\n  %26 = sc.state %25 : bool\n  %27 = sc.store %26, %23, enable=none\n  sc.output \"v_out\" %1\n  sc.output \"spike_out\" %26\n}\n",
  "graph_name": "sc_equation_neuron",
  "n_ops": 30,
  "n_inputs": 1,
  "n_outputs": 2,
  "valid": true,
  "errors": [],
  "source_present": true,
  "source_replay": "not_requested",
  "source_sha256": "2a1dba0643f5ff11a5911c5587f2fe4608ccc53837c8f41559f31c837ecb613f",
  "source_graph_sha256": "afb10d1ff452f382940ee1c68e94a00c8b6e8c9a84f8aa630439c008e9550781",
  "clocks_per_timestep": 1,
  "pipeline_registers": 0,
  "output_widths": [
    {
      "name": "v_out",
      "width": 16
    },
    {
      "name": "spike_out",
      "width": 1
    }
  ],
  "systemverilog": null,
  "studio_job_receipt": {
    "schema_version": "studio.job-receipt.v1",
    "job_id": "sj_28a953cb3177400a",
    "kind": "compiler",
    "status": "completed",
    "artifacts": [
      {
        "relative_path": "equation-ir/result.json",
        "sha256": "8116cbb42a7b9a081012fac2220dc94cabcf7a5a4e6dbdb6ac28248b3c38a939",
        "size_bytes": 3534
      },
      {
        "relative_path": "equation-ir/evidence.json",
        "sha256": "100a4be32924a9fc6355b482203f27c5f027ad36bc780f020ad7c38c154fb03e",
        "size_bytes": 1114
      }
    ]
  }
};

const emitted: SVEmitResponse = {
  "operation": "emit-sv",
  "ir_text": "sc.graph @sc_equation_neuron {\n  sc.source v1 source=2a1dba0643f5ff11a5911c5587f2fe4608ccc53837c8f41559f31c837ecb613f graph=afb10d1ff452f382940ee1c68e94a00c8b6e8c9a84f8aa630439c008e9550781 payload=7b226d6f64656c223a7b22626173655f73746174655f756e697473223a7b7d2c22636f6e7374616e7473223a5b5d2c22646574656374696f6e223a226c6576656c222c22646973706c61795f73746174655f756e697473223a7b7d2c226474223a302e352c226571756174696f6e73223a5b5b2276222c2249225d5d2c22696e697469616c5f7374617465223a5b5b2276222c305d5d2c226d6574686f64223a2265756c6572222c22706172616d6574657273223a5b5d2c2270726f626162696c6974795f65787072657373696f6e223a6e756c6c2c2272616e646f6d5f73747265616d223a302c22726174655f65787072657373696f6e223a6e756c6c2c227265736574223a5b5d2c22726e675f73656564223a34343235372c2272756e74696d655f756e697473223a7b7d2c227370696b655f76616c7565223a6e756c6c2c227375627374657073223a312c227468726573686f6c64223a6e756c6c2c227468726573686f6c645f726e67223a226c6673723136222c22756e697473223a226e6f6e65227d2c226d6f64756c655f6e616d65223a2273635f6571756174696f6e5f6e6575726f6e222c226e756d65726963223a7b22646174615f7769647468223a31362c226672616374696f6e223a382c2266756e6374696f6e5f6c6f776572696e67223a226c75742d7631222c226f766572666c6f77223a227361747572617465222c2272616e646f6d5f707265636973696f6e223a227374616e64617264222c22726f756e64696e67223a227472756e63617465222c227369676e6564223a747275657d2c22706970656c696e65223a7b22706f696e7473223a5b5d2c22737461676573223a307d2c22736368656d61223a2273632d6571756174696f6e2d736f757263652e7631222c2273656d616e74696373223a7b226469766973696f6e223a22656e636f6465642d6e6f6e7a65726f2d6f722d736f757263652d6775617264222c226576656e745f6f766572666c6f77223a227361747572617465222c227265736574223a226173796e6368726f6e6f75732d6163746976652d6c6f772d696e697469616c2d7374617465222c22726e67223a226d6f64656c2d7365656436342d73747265616d33322d736974652d636f756e746572222c2273746174655f6f766572666c6f77223a227361747572617465227d7d\n  %0 = sc.constant 0 : i16\n  %1 = sc.state %0 : i16\n  %2 = sc.input \"I_t\" : i16\n  %3 = sc.word resize %2 : i32\n  %4 = sc.word resize %3 : i32\n  %5 = sc.constant 128 : i32\n  %6 = sc.word mul %4, %5 : i32\n  %7 = sc.constant 8 : i32\n  %8 = sc.word ashr %6, %7 : i32\n  %9 = sc.word resize %8 : i16\n  %10 = sc.word resize %1 : i17\n  %11 = sc.word resize %9 : i17\n  %12 = sc.word add %10, %11 : i17\n  %13 = sc.constant -32768 : i17\n  %14 = sc.constant 32767 : i17\n  %15 = sc.word cmp_ge %12, %13 : bool\n  %16 = sc.word cmp_le %12, %14 : bool\n  %17 = sc.word select %16, %12, %14 : i17\n  %18 = sc.word select %15, %17, %13 : i17\n  %19 = sc.word resize %18 : i16\n  %20 = sc.constant 0 : i1\n  %21 = sc.word cmp_eq %20, %20 : bool\n  %22 = sc.constant i64(0) : bool\n  %23 = sc.word select %21, %22, %22 : bool\n  %24 = sc.store %1, %19, enable=%21\n  %25 = sc.constant i64(0) : bool\n  %26 = sc.state %25 : bool\n  %27 = sc.store %26, %23, enable=none\n  sc.output \"v_out\" %1\n  sc.output \"spike_out\" %26\n}\n",
  "graph_name": "sc_equation_neuron",
  "n_ops": 30,
  "n_inputs": 1,
  "n_outputs": 2,
  "valid": true,
  "errors": [],
  "source_present": true,
  "source_replay": "passed",
  "source_sha256": "2a1dba0643f5ff11a5911c5587f2fe4608ccc53837c8f41559f31c837ecb613f",
  "source_graph_sha256": "afb10d1ff452f382940ee1c68e94a00c8b6e8c9a84f8aa630439c008e9550781",
  "clocks_per_timestep": 1,
  "pipeline_registers": 0,
  "output_widths": [
    {
      "name": "v_out",
      "width": 16
    },
    {
      "name": "spike_out",
      "width": 1
    }
  ],
  "systemverilog": "// Auto-generated by SC-NeuroCore IR Compiler v3.0\n// Source graph: sc_equation_neuron\n// Do not edit — regenerate from IR source.\n\n`timescale 1ns / 1ps\n\nmodule sc_equation_neuron (\n    input wire clk,\n    input wire rst_n,\n    input wire [15:0] I_t,\n    output wire [15:0] v_out,\n    output wire spike_out\n);\n\n    localparam signed [15:0] c0 = 16'sd0;\n    reg [15:0] v1;\n    wire [31:0] v3;\n    wire [31:0] v4;\n    localparam signed [31:0] c5 = 32'sd128;\n    wire [31:0] v6;\n    localparam signed [31:0] c7 = 32'sd8;\n    wire [31:0] v8;\n    wire [15:0] v9;\n    wire [16:0] v10;\n    wire [16:0] v11;\n    wire [16:0] v12;\n    localparam signed [16:0] c13 = -17'sd32768;\n    localparam signed [16:0] c14 = 17'sd32767;\n    wire [0:0] v15;\n    wire [0:0] v16;\n    wire [16:0] v17;\n    wire [16:0] v18;\n    wire [15:0] v19;\n    localparam signed [0:0] c20 = 1'sd0;\n    wire [0:0] v21;\n    localparam [0:0] c22 = 1'd0;\n    wire [0:0] v23;\n    localparam [0:0] c25 = 1'd0;\n    reg [0:0] v26;\n\n    assign v3 = $signed(I_t);\n    assign v4 = $signed(v3);\n    assign v6 = $signed(v4) * $signed(c5);\n    assign v8 = $signed(v6) >>> $unsigned(c7);\n    assign v9 = $signed(v8);\n    assign v10 = $signed(v1);\n    assign v11 = $signed(v9);\n    assign v12 = $signed(v10) + $signed(v11);\n    assign v15 = $signed(v12) >= $signed(c13);\n    assign v16 = $signed(v12) <= $signed(c14);\n    assign v17 = v16 ? v12 : c14;\n    assign v18 = v15 ? v17 : c13;\n    assign v19 = $signed(v18);\n    assign v21 = $signed(c20) == $signed(c20);\n    assign v23 = v21 ? c22 : c22;\n    always @(posedge clk or negedge rst_n) begin\n        if (!rst_n) v1 <= c0;\n        else if (v21) v1 <= v19;\n    end\n    always @(posedge clk or negedge rst_n) begin\n        if (!rst_n) v26 <= c25;\n        else v26 <= v23;\n    end\n    assign v_out = v1;\n    assign spike_out = v26;\n\nendmodule\n",
  "studio_job_receipt": {
    "schema_version": "studio.job-receipt.v1",
    "job_id": "sj_11f8852f3b81c6e2",
    "kind": "compiler",
    "status": "completed",
    "artifacts": [
      {
        "relative_path": "equation-ir/result.json",
        "sha256": "d39044c8e2fc858c594bf9d1f928d32e2cf4ba44189bda91303838b7355d9d1f",
        "size_bytes": 5439
      },
      {
        "relative_path": "equation-ir/evidence.json",
        "sha256": "e41ac1f496b96eab4cd82bb3205db783e5e341d5f7b740aa652f25b2326ed42c",
        "size_bytes": 1116
      }
    ]
  }
};

/**
 * Copy an actual native build receipt for state-transition tests.
 *
 * @param overrides - Explicit changes exercised by the transition under test.
 * @returns Independent native result data.
 */
export function nativeIRBuildResult(overrides: Partial<IRBuildResponse> = {}): IRBuildResponse {
  return { ...structuredClone(built), ...overrides };
}

/**
 * Copy an actual source-replayed native SystemVerilog result.
 *
 * @param overrides - Explicit changes exercised by the transition under test.
 * @returns Independent native result and output contract.
 */
export function nativeSVEmitResult(overrides: Partial<SVEmitResponse> = {}): SVEmitResponse {
  return { ...structuredClone(emitted), ...overrides };
}

const invalid: IRVerifyResponse = {
  "operation": "verify",
  "ir_text": "sc.graph @invalid_clock_port {\n  %0 = sc.input \"clk\" : bool\n  sc.output \"value\" %0\n}\n",
  "graph_name": "invalid_clock_port",
  "n_ops": 2,
  "n_inputs": 1,
  "n_outputs": 1,
  "valid": false,
  "errors": [
    "op[0]: HDL port identifier uses a reserved generated namespace: \"clk\""
  ],
  "source_present": false,
  "source_replay": "unavailable",
  "source_sha256": null,
  "source_graph_sha256": null,
  "clocks_per_timestep": null,
  "pipeline_registers": null,
  "output_widths": null,
  "systemverilog": null,
  "studio_job_receipt": {
    "schema_version": "studio.job-receipt.v1",
    "job_id": "sj_d68bf45918b3ad35",
    "kind": "compiler",
    "status": "completed",
    "artifacts": [
      {
        "relative_path": "equation-ir/result.json",
        "sha256": "3248f1acd157f0cc10f9497b0b3f23fc61dcfe429c264e5925c60c0f019c7291",
        "size_bytes": 572
      },
      {
        "relative_path": "equation-ir/evidence.json",
        "sha256": "a179f67709361558124830ab272be3a48de1f82e6e8b5765dbeef552a8dfdc8e",
        "size_bytes": 1114
      }
    ]
  }
};

/**
 * Copy actual native diagnostics for a reserved implicit clock port.
 *
 * @returns A real structural rejection without source replay availability.
 */
export function nativeIRInvalidResult(): IRVerifyResponse {
  return structuredClone(invalid);
}
