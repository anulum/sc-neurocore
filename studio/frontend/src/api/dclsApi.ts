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

export const fetchDclsInfo = () => get<DclsInfo>("/dcls/info");

export const fetchDclsBenchmark = () => get<DclsBenchmark>("/dcls/benchmark");

export const evaluateDcls = (body: DclsEvaluateBody) =>
  post<DclsEvaluation>("/dcls/evaluate", body);

