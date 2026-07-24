// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: models endpoints.
import { post, get } from "./http";
import type {
  NeuronTemplate,
  ModelSummary,
  ModelDetail,
  ModelFacets,
  ModelDoc,
  PresetSummary,
  ModelScanJobReceipt,
  ModelScanResponse,
} from "./types";

export const fetchTemplates = () => get<NeuronTemplate[]>("/templates");

export const fetchModels = () => get<ModelSummary[]>("/models");

export const fetchModelDetail = (name: string) => get<ModelDetail>(`/models/${name}`);

export const fetchModelFacets = () => get<ModelFacets>("/models/facets");

export const fetchModelDoc = (name: string) =>
  get<ModelDoc>(`/models/${encodeURIComponent(name)}/doc`);

export const fetchPresets = () => get<PresetSummary[]>("/presets");

export const fetchPreset = (id: string) => get<Record<string, unknown>>(`/presets/${id}`);

export const fetchModelScan = () => get<ModelScanResponse>("/models/scan");

export const submitModelScanJob = () =>
  post<ModelScanJobReceipt>("/models/scan/jobs", {});
