// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio selected-model compile configuration

import type { ModelCompileRequest, ModelCosimRequest, ModelDetail } from "./api/client";

export interface StudioModelCompileInput {
  dt: number;
  integrator: string;
  modelDetail: ModelDetail | null;
  modelParams: Record<string, number>;
  qFormat: string;
  selectedModelName: string;
}

/** Build the typed model compile request without leaking state-variable initials as parameters. */
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
  if (!configuration.q_formats.includes(qFormat)) {
    throw new Error(`Q-format ${qFormat} is not offered for the selected model.`);
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

/** Build a co-simulation request over the exact same selected compiler configuration. */
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
