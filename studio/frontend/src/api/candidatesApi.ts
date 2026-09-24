// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: candidate model packages.
import { post } from "./http";

/** One problem with a candidate, located by a JSON pointer. */
export interface CandidateDiagnostic {
  location: string;
  message: string;
}

/** What `POST /api/candidates/validate` answers. */
export interface CandidateValidation {
  schema_version: string;
  valid: boolean;
  candidate_sha256: string | null;
  diagnostics: CandidateDiagnostic[];
}

/** One value (state or parameter) compared with the parent's. */
export interface CandidateValueRow {
  name: string;
  status: "added" | "removed" | "changed" | "unchanged";
  parent?: number;
  candidate?: number;
}

/** One equation compared with the parent's, as text and as mathematics. */
export interface CandidateEquationRow {
  variable: string;
  status: "added" | "removed" | "unchanged" | "equivalent" | "changed" | "undecided" | "not_comparable";
  parent?: string;
  candidate?: string;
  difference?: string;
  reason?: string;
}

/** What `POST /api/candidates/diff` answers. */
export interface CandidateDiff {
  schema_version: string;
  parent: string | null;
  status: "compared" | "no_parent" | "parent_has_no_schema";
  parent_schema?: string;
  state?: CandidateValueRow[];
  parameters?: CandidateValueRow[];
  dynamics?: CandidateEquationRow[];
  reset?: CandidateEquationRow[];
  threshold?: { status: string; difference?: string; reason?: string };
  integration?: { status: string; changed_fields: string[] };
}

/** What `POST /api/candidates/simulate` answers. */
export interface CandidateRun {
  candidate: string;
  candidate_sha256: string;
  units: { current: string; time: string };
  steps: number;
  current: number;
  profile: { method: string; dt: number; time_unit: string };
  spike_count: number;
  spike_steps: number[];
  final_state: Record<string, number> | null;
  diverged_at_step: number | null;
  divergence: string | null;
  sample_every: number;
  trace: Record<string, number[]>;
}

/** One reference test's outcome in a review packet. */
export interface CandidateTestResult {
  name: string;
  passed: boolean;
  diverged_at_step: number | null;
  checks: { quantity: string; observed: number | null; held: boolean }[];
}

/** What `POST /api/candidates/review-packet` answers. */
export interface CandidateReviewPacket {
  schema_version: string;
  candidate_sha256: string;
  packet_sha256: string;
  reference_tests: CandidateTestResult[];
  reference_tests_passed: boolean;
  not_established: string[];
}

/**
 * Report every problem with a candidate package.
 *
 * @param candidate - The parsed package.
 * @returns The located diagnostics; an invalid candidate is still HTTP 200.
 */
export const validateCandidate = (candidate: unknown) =>
  post<CandidateValidation>("/candidates/validate", { candidate });

/**
 * Diff a valid candidate against the catalogue model it names as parent.
 *
 * @param candidate - The parsed package.
 * @returns The diff; an invalid candidate is refused with HTTP 422.
 */
export const diffCandidate = (candidate: unknown) =>
  post<CandidateDiff>("/candidates/diff", { candidate });

/**
 * Simulate a valid candidate under its own numerical profile.
 *
 * @param candidate - The parsed package.
 * @param current - Constant injected current, in the candidate's current unit.
 * @param steps - Integration steps to take.
 * @returns The run.
 */
export const simulateCandidate = (candidate: unknown, current: number, steps: number) =>
  post<CandidateRun>("/candidates/simulate", { candidate, current, steps });

/**
 * Run a valid candidate's reference tests and assemble its review packet.
 *
 * @param candidate - The parsed package.
 * @returns The packet, bound under one digest.
 */
export const candidateReviewPacket = (candidate: unknown) =>
  post<CandidateReviewPacket>("/candidates/review-packet", { candidate });
