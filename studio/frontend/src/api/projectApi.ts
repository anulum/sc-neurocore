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
  ProjectBranchResponse,
  PipelineResult,
} from "./types";

/**
 * Save a workspace as a new revision.
 *
 * `expectedRevision` is the revision the caller loaded and edited; `null` says
 * "this workspace is new". Saving from a revision that is no longer current is
 * refused with HTTP 409 rather than overwriting the other editor's work.
 *
 * @param name - The workspace's name.
 * @param state - The workspace to store.
 * @param expectedRevision - The revision this edit started from, or `null`.
 * @returns The revision the server wrote.
 */
export const saveProject = (
  name: string,
  state: Record<string, unknown>,
  expectedRevision: number | null = null,
) => post<ProjectSaveResponse>("/project/save", { name, state, expected_revision: expectedRevision });

/**
 * Keep an edit a save conflict refused, as a branch of its own.
 *
 * A stale save is refused so it cannot overwrite the other editor's work, which
 * leaves the refused edit only in this browser. Sending it here stores it as the
 * first revision of its own workspace, named for the revision it diverged from,
 * so it survives a closed tab and can be compared against what won.
 *
 * @param name - The workspace whose save was refused.
 * @param state - The refused workspace state, exactly as held here.
 * @param baseRevision - The revision this edit started from.
 * @returns The branch that now holds the edit.
 */
export const branchRefusedEdit = (
  name: string,
  state: Record<string, unknown>,
  baseRevision: number,
) =>
  post<ProjectBranchResponse>(`/project/${name}/branch-refused-edit`, {
    state,
    base_revision: baseRevision,
  });

/**
 * Load a workspace revision; `null` reads the current one.
 *
 * @param name - The workspace's name.
 * @param revision - The revision to read, or `null` for the current one.
 * @returns The stored workspace.
 */
export const loadProject = (name: string, revision: number | null = null) =>
  get<Record<string, unknown>>(
    revision === null
      ? `/project/load/${name}`
      : `/project/load/${name}?revision=${encodeURIComponent(String(revision))}`,
  );

/**
 * List every stored revision of one workspace, oldest first.
 *
 * @param name - The workspace's name.
 * @returns The revisions and what each was saved from.
 */
export const listProjectRevisions = (name: string) =>
  get<ProjectRevisionList>(`/project/${name}/revisions`);

/**
 * List the stored workspaces.
 *
 * @returns One summary per workspace.
 */
export const listProjects = () => get<ProjectSummary[]>("/project/list");

/**
 * Move a workspace to the recoverable trash.
 *
 * It goes to the trash rather than away: {@link listDeletedProjects} and
 * {@link restoreProject} bring it back. `DELETE` has no helper in the
 * transport, which is why this calls `fetch` directly.
 *
 * @param name - The workspace's name.
 * @returns The name the server deleted.
 */
export const deleteProject = (name: string) =>
  fetch(`/api/project/${name}`, { method: "DELETE", headers: authHeaders() }).then((r) =>
    json<{ deleted: string }>(r),
  );

/**
 * List the workspaces waiting in the recoverable trash, newest first.
 *
 * @returns The deleted workspaces and the tokens that restore them.
 */
export const listDeletedProjects = () =>
  get<{ deleted: DeletedProjectSummary[] }>("/project/deleted");

/**
 * Bring one deleted workspace back under its original name.
 *
 * @param token - The restore token from {@link listDeletedProjects}.
 * @returns The name restored and the revision it came back at.
 */
export const restoreProject = (token: string) =>
  post<{ restored: string; revision: number | null }>("/project/restore", { token });

/**
 * Take a graph through the whole pipeline to a target in one request.
 *
 * @param graph - The graph to run through.
 * @param target - The device family to end at.
 * @returns Each step's outcome, and the step that failed if one did.
 */
export const runPipeline = (graph: NetworkGraph, target: string) =>
  post<PipelineResult>("/pipeline/run", { graph, target });

/** One review comment, checked against the revision it was written on. */
export interface ReviewComment {
  comment_id: string;
  revision: number;
  state_sha256: string;
  author: string;
  created_at: number;
  body: string;
  reply_to: string | null;
  /** `matches`, or `changed`/`missing` when the revision no longer is what was reviewed. */
  revision_status: "matches" | "changed" | "missing";
}

/**
 * List a workspace's review comments on one revision.
 *
 * @param name - The workspace's name.
 * @param revision - The revision.
 * @returns The comments, each with its revision's status.
 */
export const listReviewComments = (name: string, revision: number) =>
  get<{ comments: ReviewComment[] }>(
    `/project/${encodeURIComponent(name)}/comments?revision=${encodeURIComponent(String(revision))}`,
  );

/**
 * Comment on one immutable revision.
 *
 * @param name - The workspace's name.
 * @param revision - The revision reviewed.
 * @param body - The comment.
 * @param replyTo - The comment answered, on the same revision, or `null`.
 * @returns The stored comment.
 */
export const addReviewComment = (name: string, revision: number, body: string, replyTo: string | null) =>
  post<ReviewComment>(
    `/project/${encodeURIComponent(name)}/revisions/${encodeURIComponent(String(revision))}/comments`,
    { body, reply_to: replyTo },
  );
