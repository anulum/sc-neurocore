// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Candidate workbench: reading drafts and server answers

/**
 * What the candidate panel needs to turn text and server answers into a view.
 *
 * A draft is kept as the author typed it, valid JSON or not, so saving a
 * workspace never rewrites or loses a half-finished candidate. Only an action
 * that sends the candidate parses it, and a draft that is not JSON is reported
 * where it breaks rather than sent.
 */

import type {
  CandidateDiff,
  CandidateEquationRow,
  CandidateValidation,
  CandidateValueRow,
} from "./api/candidatesApi";
import { StudioRequestError } from "./api/http";

/** A draft read as JSON, or the reason it could not be. */
export type CandidateParse =
  | { ok: true; document: unknown }
  | { ok: false; message: string };

/**
 * Read a draft as the JSON document it should be.
 *
 * @param text - The draft as typed.
 * @returns The document, or the parser's own message.
 */
export function parseCandidateText(text: string): CandidateParse {
  if (text.trim() === "") {
    return { ok: false, message: "The draft is empty: import a candidate package or write one." };
  }
  try {
    return { ok: true, document: JSON.parse(text) as unknown };
  } catch (error) {
    return {
      ok: false,
      message: `The draft is not JSON: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

/**
 * Read the validation out of a refusal, when the refusal is one.
 *
 * The routes that act on a candidate refuse an invalid one with HTTP 422 and
 * the same located diagnostics the validation route returns.
 *
 * @param error - What a candidate request rejected with.
 * @returns The validation, or `null` for any other failure.
 */
export function candidateRefusal(error: unknown): CandidateValidation | null {
  if (!(error instanceof StudioRequestError) || error.status !== 422) return null;
  const detail = error.detail as { reason?: unknown; validation?: unknown } | null;
  if (detail === null || typeof detail !== "object" || detail.reason !== "invalid_candidate") {
    return null;
  }
  return detail.validation as CandidateValidation;
}

/** One line of a diff as the panel lists it. */
export interface CandidateDiffLine {
  section: string;
  name: string;
  status: string;
  detail: string;
}

/**
 * Describe one value row.
 *
 * @param row - The row.
 * @returns What changed, in words.
 */
function valueDetail(row: CandidateValueRow): string {
  if (row.status === "added") return `added: ${String(row.candidate)}`;
  if (row.status === "removed") return `removed (was ${String(row.parent)})`;
  return `${String(row.parent)} → ${String(row.candidate)}`;
}

/**
 * Describe one equation row.
 *
 * @param row - The row.
 * @returns What changed, in words.
 */
function equationDetail(row: CandidateEquationRow): string {
  if (row.status === "added") return `added: ${row.candidate ?? ""}`;
  if (row.status === "removed") return `removed (was ${row.parent ?? ""})`;
  if (row.status === "changed") return `candidate − parent = ${row.difference ?? ""}`;
  if (row.status === "equivalent") return "rewritten, mathematically equal";
  return row.reason ?? "";
}

/**
 * List what a candidate changes, leaving out what it keeps.
 *
 * @param diff - The server's diff.
 * @returns One line per change, section by section; empty when nothing changed.
 */
export function candidateDiffLines(diff: CandidateDiff): CandidateDiffLine[] {
  const lines: CandidateDiffLine[] = [];
  for (const [section, rows] of [["state", diff.state], ["parameters", diff.parameters]] as const) {
    for (const row of rows ?? []) {
      if (row.status !== "unchanged") {
        lines.push({ section, name: row.name, status: row.status, detail: valueDetail(row) });
      }
    }
  }
  for (const [section, rows] of [["dynamics", diff.dynamics], ["reset", diff.reset]] as const) {
    for (const row of rows ?? []) {
      if (row.status !== "unchanged") {
        lines.push({ section, name: row.variable, status: row.status, detail: equationDetail(row) });
      }
    }
  }
  if (diff.threshold !== undefined && diff.threshold.status !== "unchanged") {
    lines.push({
      section: "threshold",
      name: "condition",
      status: diff.threshold.status,
      detail: diff.threshold.difference ?? diff.threshold.reason ?? "",
    });
  }
  if (diff.integration !== undefined && diff.integration.status !== "unchanged") {
    lines.push({
      section: "integration",
      name: diff.integration.changed_fields.join(", "),
      status: "changed",
      detail: "numerical profile differs from the parent's",
    });
  }
  return lines;
}

/**
 * State why a diff has no lines to show, when it has none.
 *
 * @param diff - The server's diff.
 * @returns The sentence, or `null` when the diff was compared.
 */
export function candidateDiffNotice(diff: CandidateDiff): string | null {
  if (diff.status === "no_parent") return "The candidate names no parent, so there is nothing to diff against.";
  if (diff.status === "parent_has_no_schema") {
    return `${diff.parent ?? "The parent"} has no canonical schema to compare with.`;
  }
  return null;
}

/**
 * Name a file for a candidate or its review packet.
 *
 * @param document - The parsed candidate.
 * @param kind - What the file holds.
 * @returns A file name built from the candidate's name, or a generic one.
 */
export function candidateFileName(document: unknown, kind: "candidate" | "review"): string {
  const name = (document as { name?: unknown } | null)?.name;
  const stem = typeof name === "string" && /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(name) ? name : "candidate";
  return `${stem}.${kind}.json`;
}
