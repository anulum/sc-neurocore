// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Authored equation native compiler wire contracts

import type { StudioJobReceipt } from "./types";

/** Maintained integration methods, preserving authored Gauss–Seidel order. */
export type EquationIntegrationMethod = "euler" | "rk4" | "exp_euler" | "map" | "gauss_seidel";
/** Supported event-detection modes; individual combinations may refuse lowering. */
export type EquationDetection = "level" | "crossing" | "poisson" | "escape_rate";
/** Threshold generator selected explicitly by the authored source. */
export type EquationThresholdRng = "lfsr16" | "philox";
/** Native integrated-state overflow semantics. */
export type EquationOverflow = "saturate" | "wrap" | "trap";
/** Native product rounding semantics. */
export type EquationRounding = "truncate" | "nearest" | "bankers";
/** Maintained function lowering profiles, with opt-in structured functions. */
export type EquationFunctionLowering = "lut-v1" | "v2";
/** Native normal-draw precision profile. */
export type EquationRandomPrecision = "standard" | "exact";

/** Scalar magnitude and optional physical unit under strict dimensional validation. */
export interface EquationValue {
  value: number;
  unit: string | null;
}
/** A declaration whose position is preserved through worker JSON serialization. */
export interface NamedEquationValue extends EquationValue {
  name: string;
}
/** One ordered state derivative or reset expression. */
export interface EquationExpression {
  name: string;
  expression: string;
}
/** All serializable constructor inputs of an authored native equation model. */
export interface AuthoredEquationModel {
  equations: EquationExpression[];
  initial_state: NamedEquationValue[];
  parameters: NamedEquationValue[];
  constants: NamedEquationValue[];
  reset: EquationExpression[];
  threshold: string | null;
  dt: EquationValue;
  method: EquationIntegrationMethod;
  detection: EquationDetection;
  substeps: number;
  rate_expression: string | null;
  probability_expression: string | null;
  spike_value: string | null;
  threshold_rng: EquationThresholdRng;
  /** Canonical unsigned 64-bit decimal text; never converted to a JS number. */
  rng_seed: string;
  random_stream: number;
  units: "none" | "strict";
  input_unit: EquationValue | null;
}
/** Exact native scalar geometry, registration and function-lowering choices. */
export interface EquationIRProfile {
  module_name: string;
  /** Total signed scalar bits, including fractional bits. */
  data_width: number;
  fraction: number;
  signed: boolean;
  overflow: EquationOverflow;
  rounding: EquationRounding;
  pipeline_stages: number;
  pipeline_points: string[];
  function_lowering: EquationFunctionLowering;
  random_precision: EquationRandomPrecision;
}
/** Versioned request accepted by the real bounded equation compiler worker. */
export interface EquationIRRequest {
  schema_version: "studio.equation-ir-request.v1";
  model: AuthoredEquationModel;
  profile: EquationIRProfile;
}
/** Actual native output name and physical width in declaration order. */
export interface EquationIRPort {
  name: string;
  width: number;
}
/** Native graph facts; structural verification and semantic replay are separate claims. */
export interface EquationIRResult {
  operation: "build" | "verify" | "replay" | "emit-sv";
  ir_text: string;
  graph_name: string;
  n_ops: number;
  n_inputs: number;
  n_outputs: number;
  valid: boolean;
  errors: string[];
  source_present: boolean;
  source_replay: "passed" | "unavailable" | "not_requested";
  source_sha256: string | null;
  source_graph_sha256: string | null;
  clocks_per_timestep: number | null;
  pipeline_registers: number | null;
  output_widths: EquationIRPort[] | null;
  systemverilog: string | null;
  studio_job_receipt: StudioJobReceipt;
}
/** Built native graph with actual cadence and output contract. */
export interface IRBuildResponse extends EquationIRResult {
  operation: "build";
  valid: true;
  errors: [];
  clocks_per_timestep: number;
  pipeline_registers: number;
  output_widths: EquationIRPort[];
  systemverilog: null;
}
/** Structural inspection without a semantic replay claim. */
export interface IRVerifyResponse extends EquationIRResult {
  operation: "verify";
  systemverilog: null;
}
/** Reconstructed source and graph with successful semantic equality. */
export interface IRReplayResponse extends EquationIRResult {
  operation: "replay";
  source_replay: "passed";
  systemverilog: null;
}
/** Actual native RTL, subject to the request's explicit source policy. */
export interface SVEmitResponse extends EquationIRResult {
  operation: "emit-sv";
  systemverilog: string;
}
