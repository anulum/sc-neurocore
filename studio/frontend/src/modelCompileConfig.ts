// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio selected-model compile configuration

import type {
  ModelCompileConfiguration,
  ModelCompileRequest,
  ModelCosimRequest,
  ModelDetail,
} from "./api/client";

/** What the compile needs: the model, and the settings the reader chose. */
export interface StudioModelCompileInput {
  dt: number;
  integrator: string;
  modelDetail: ModelDetail | null;
  modelParams: Record<string, number>;
  qFormat: string;
  selectedModelName: string;
}

/**
 * Build the request that compiles the selected model to RTL.
 *
 * The parameters are taken from the model's own declared parameter list, each
 * either as the reader set it or at the model's default. That is deliberate:
 * reading the panel's parameter map directly would carry state-variable
 * initial values across as if they were parameters, and the compiler would
 * take them.
 *
 * The integrator and Q-format are checked against what the model declares
 * rather than corrected, because a compile at an undeclared setting produces
 * RTL nobody has validated.
 *
 * @param input - The selected model and the settings the reader chose.
 * @returns The compile request.
 * @throws {Error} When no model is selected, when the model has no canonical
 *   RTL path, or when a setting is not one the model declares. Each message
 *   names what the reader has to change.
 */
export function modelCompileRequest(input: StudioModelCompileInput): ModelCompileRequest {
  const configuration = input.modelDetail?.compile_configuration;
  if (input.modelDetail === null || input.selectedModelName.length === 0) {
    throw new Error("Choose a catalogue model before compiling RTL.");
  }
  if (configuration === null || configuration === undefined) {
    throw new Error("The selected model has no canonical schema-backed RTL path.");
  }
  const integrator = input.integrator || configuration.default_integrator;
  if (!configuration.integrators.includes(integrator)) {
    throw new Error(`Integrator ${integrator} is not declared for the selected model.`);
  }
  const qFormat = input.qFormat || configuration.default_q_format;
  if (qFormat === null) {
    throw new Error(
      `No Q-format Studio compiles at can hold the selected model. ${qFormatRefusals(configuration).map((item) => item.refusal).join(" ")}`,
    );
  }
  if (!configuration.q_formats.includes(qFormat)) {
    const refusal = configuration.numeric_contracts[qFormat]?.refusal;
    throw new Error(
      `Q-format ${qFormat} is not offered for the selected model.${refusal ? ` ${refusal}` : ""}`,
    );
  }
  const params = Object.fromEntries(
    input.modelDetail.params.map((parameter) => [
      parameter.name,
      input.modelParams[parameter.name] ?? parameter.default,
    ]),
  );
  return {
    dt: input.dt,
    integrator,
    model_name: input.selectedModelName,
    params,
    q_format: qFormat,
  };
}

/**
 * List the candidate Q-formats the model is not representable in, and why.
 *
 * @param configuration - The model's compile configuration.
 * @returns Each refused format with the contract's reason, in candidate order.
 */
export function qFormatRefusals(
  configuration: ModelCompileConfiguration,
): { qFormat: string; refusal: string }[] {
  return Object.entries(configuration.numeric_contracts)
    .filter(([, contract]) => !contract.representable)
    .map(([qFormat, contract]) => ({ qFormat, refusal: contract.refusal }));
}

/**
 * Build the request that co-simulates the selected model against its RTL.
 *
 * It is built on top of the compile request rather than beside it, so the
 * co-simulation cannot run against a configuration the compile would refuse.
 *
 * @param input - The selected model and the settings the reader chose.
 * @param stimulus - The current to drive it with, and how many steps to run.
 * @returns The co-simulation request.
 * @throws {Error} When the compile request cannot be built, or when the chosen
 *   integrator has no co-simulation support.
 */
export function modelCosimRequest(
  input: StudioModelCompileInput,
  stimulus: { current: number; nSteps?: number },
): ModelCosimRequest {
  const compileRequest = modelCompileRequest(input);
  const supported = input.modelDetail?.compile_configuration?.cosim_integrators ?? [];
  if (!supported.includes(compileRequest.integrator)) {
    throw new Error(
      `Integrator ${compileRequest.integrator} has no bit-exact selected-model co-simulation path.`,
    );
  }
  return {
    ...compileRequest,
    current: stimulus.current,
    n_steps: stimulus.nSteps ?? 128,
  };
}
