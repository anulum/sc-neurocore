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
  DeletedProjectSummary,
  NetworkGraph,
  ProjectRevisionList,
  ProjectSummary,
  ProjectSaveResponse,
  PipelineResult,
} from "./types";

/**
 * Save a workspace as a new revision.
 *
 * `expectedRevision` is the revision the caller loaded and edited; `null` says
 * "this workspace is new". Saving from a revision that is no longer current is
 * refused with HTTP 409 rather than overwriting the other editor's work.
 */
export const saveProject = (
  name: string,
  state: Record<string, unknown>,
  expectedRevision: number | null = null,
) => post<ProjectSaveResponse>("/project/save", { name, state, expected_revision: expectedRevision });

/** Load a workspace revision; `null` reads the current one. */
export const loadProject = (name: string, revision: number | null = null) =>
  get<Record<string, unknown>>(
    revision === null
      ? `/project/load/${name}`
      : `/project/load/${name}?revision=${encodeURIComponent(String(revision))}`,
  );

/** List every stored revision of one workspace, oldest first. */
export const listProjectRevisions = (name: string) =>
  get<ProjectRevisionList>(`/project/${name}/revisions`);

export const listProjects = () => get<ProjectSummary[]>("/project/list");

export const deleteProject = (name: string) =>
  fetch(`/api/project/${name}`, { method: "DELETE", headers: authHeaders() }).then((r) =>
    json<{ deleted: string }>(r),
  );

/** List the workspaces waiting in the recoverable trash, newest first. */
export const listDeletedProjects = () =>
  get<{ deleted: DeletedProjectSummary[] }>("/project/deleted");

/** Bring one deleted workspace back under its original name. */
export const restoreProject = (token: string) =>
  post<{ restored: string; revision: number | null }>("/project/restore", { token });

export const runPipeline = (graph: NetworkGraph, target: string) =>
  post<PipelineResult>("/pipeline/run", { graph, target });
