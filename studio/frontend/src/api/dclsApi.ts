// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: dcls endpoints.
import { post, get } from "./http";
import type {
  DclsInfo,
  DclsEvaluation,
  DclsEvaluateBody,
  DclsBenchmark,
} from "./types";

/**
 * Ask what the delay-and-current-learning-synapse backend supports.
 *
 * @returns The backend's capabilities and its current configuration.
 */
export const fetchDclsInfo = () => get<DclsInfo>("/dcls/info");

/**
 * Read the recorded DCLS benchmark, which the server measured, not this call.
 *
 * @returns The stored benchmark figures and their provenance.
 */
export const fetchDclsBenchmark = () => get<DclsBenchmark>("/dcls/benchmark");

/**
 * Run one DCLS evaluation on the server.
 *
 * @param body - The kernel, the delays and the input to evaluate.
 * @returns The evaluation the server ran.
 */
export const evaluateDcls = (body: DclsEvaluateBody) =>
  post<DclsEvaluation>("/dcls/evaluate", body);
