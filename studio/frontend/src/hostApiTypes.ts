// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — finite hosted runtime contracts

/** Closed local scientific operations; no hardware or identity operation is included. */
export const HOST_OPERATIONS = ["simulate", "fi_curve", "bifurcation", "heatmap", "sensitivity",
  "ir.build", "ir.verify", "ir.replay", "ir.emit-sv"] as const;
/** One closed operation supported by the installed local adapter. */
export type HostOperation = typeof HOST_OPERATIONS[number];
/** Protocol shared with the host's fixed local runtime service. */
export const HOST_API_PROTOCOL = "sc-neurocore.host-api.v1";
/** A sealed artifact's digest and measured byte count, without a backend path. */
export interface HostArtifact { digest: string; bytes: number; result: boolean }
/** A job admitted by this panel; transport completion does not attest compiler acceptance. */
export interface HostJob {
  jobId: string;
  kind: "analysis" | "compiler";
  status: string;
  artifacts: HostArtifact[];
}
/** Exact validated JSON text and its SHA-256 binding. */
export interface HostJson { json: string; digest: string; bytes: number }
/** Only these request shapes can cross the hosted compute bridge. */
export type HostApiRequest =
  | { action: "describe" | "models" | "templates" }
  | { action: "submit"; operation: HostOperation; json: string }
  | { action: "status" | "cancel" | "result"; jobId: string }
  | { action: "artifact" | "download"; jobId: string; digest: string };
