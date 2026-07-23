// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: project endpoints.
import { authHeaders, get, json, post } from "./http";
import type {
  NetworkGraph,
  ProjectSummary,
  ProjectSaveResponse,
  PipelineResult,
} from "./types";

export const saveProject = (name: string, state: Record<string, unknown>) =>
  post<ProjectSaveResponse>("/project/save", { name, state });

export const loadProject = (name: string) => get<Record<string, unknown>>(`/project/load/${name}`);

export const listProjects = () => get<ProjectSummary[]>("/project/list");

export const deleteProject = (name: string) =>
  fetch(`/api/project/${name}`, { method: "DELETE", headers: authHeaders() }).then((r) =>
    json<{ deleted: string }>(r),
  );

export const runPipeline = (graph: NetworkGraph, target: string) =>
  post<PipelineResult>("/pipeline/run", { graph, target });

