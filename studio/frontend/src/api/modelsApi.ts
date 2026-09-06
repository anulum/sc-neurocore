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

/**
 * List the editable ODE templates the Studio offers as starting points.
 *
 * @returns Every template, with its equations and defaults.
 */
export const fetchTemplates = () => get<NeuronTemplate[]>("/templates");

/**
 * List the model catalogue, one summary per model.
 *
 * @returns Every catalogue model, enough to browse and filter but not to run.
 */
export const fetchModels = () => get<ModelSummary[]>("/models");

/**
 * Read one model's full contract: parameters, ranges, states and provenance.
 *
 * @param name - The model's catalogue name.
 * @returns Everything needed to configure and run it.
 */
export const fetchModelDetail = (name: string) => get<ModelDetail>(`/models/${name}`);

/**
 * Read the facets the catalogue can be filtered by.
 *
 * Taken from the server rather than assembled in the browser, so a facet the
 * catalogue has stopped carrying disappears from the filters with it.
 *
 * @returns Families, maturities, firing patterns and behaviour tags.
 */
export const fetchModelFacets = () => get<ModelFacets>("/models/facets");

/**
 * Read a model's prose documentation.
 *
 * @param name - The model's catalogue name.
 * @returns The document, as the server renders it.
 */
export const fetchModelDoc = (name: string) =>
  get<ModelDoc>(`/models/${encodeURIComponent(name)}/doc`);

/**
 * List the saved experiment presets.
 *
 * @returns One summary per preset.
 */
export const fetchPresets = () => get<PresetSummary[]>("/presets");

/**
 * Read one preset's configuration.
 *
 * The shape is the request body of whatever route the preset drives, so it is
 * carried opaquely rather than narrowed here.
 *
 * @param id - The preset's identifier.
 * @returns The configuration the preset stands for.
 */
export const fetchPreset = (id: string) => get<Record<string, unknown>>(`/presets/${id}`);

/**
 * Read the last catalogue scan the server completed.
 *
 * @returns The scan's findings and the evidence they were taken under.
 */
export const fetchModelScan = () => get<ModelScanResponse>("/models/scan");

/**
 * Start a fresh catalogue scan.
 *
 * The scan runs longer than a request, so this returns a receipt and the work
 * is followed on the progress socket.
 *
 * @returns The receipt the scan is followed by.
 */
export const submitModelScanJob = () =>
  post<ModelScanJobReceipt>("/models/scan/jobs", {});
