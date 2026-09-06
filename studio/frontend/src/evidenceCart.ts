// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio session evidence cart (queue + single export)

/**
 * Session-scoped evidence cart for SNN Studio.
 *
 * Operators queue simulation and analysis (and related) run artefacts into one
 * ordered cart, then produce a single export bundle with stable SHA-256 digests
 * over canonical JSON payloads. This is distinct from server-side job evidence
 * bundles (`studio.evidence-bundle.v1`): the cart is the operator session ledger
 * before or alongside project/admin bundle export.
 *
 * Digests come from the shared evidence seal, so a payload the server sealed
 * carries the same digest here. `JSON.stringify` did not: it renders `1` where
 * Python renders `1.0`, so a cart digest and a server digest of one identical
 * run never agreed and neither could check the other.
 */

import { canonicalSealText, sealSha256 } from "./evidenceSeal";

export const EVIDENCE_CART_SCHEMA_VERSION = "studio.evidence-cart.v1" as const;

/** Artefact kinds that the cart accepts for Phase 0 operator workflows. */
export type EvidenceCartItemKind =
  | "simulation"
  | "analysis"
  | "compile"
  | "synthesis"
  | "project"
  | "other";

/** Draft used when enqueuing an artefact into the cart. */
export interface EvidenceCartArtefactDraft {
  /** Honesty / evidence class label (for example analysis or curated). */
  classification: string;
  /** Stable kind used for grouping and guided-flow attachment. */
  kind: EvidenceCartItemKind;
  /** Human-readable label shown in the cart UI. */
  label: string;
  /** Payload that is digested and exported (must be JSON-serialisable). */
  payload: unknown;
  /** Optional model or source name for discoverability. */
  sourceName?: string;
}

/** One queued cart item with identity and queue timestamp. */
export interface EvidenceCartItem extends EvidenceCartArtefactDraft {
  /** Opaque item id (unique within the cart). */
  id: string;
  /** ISO-8601 UTC timestamp when the item was queued. */
  queuedAtUtc: string;
}

/** Ordered session cart state. */
export interface EvidenceCart {
  items: readonly EvidenceCartItem[];
  schema_version: typeof EVIDENCE_CART_SCHEMA_VERSION;
}

/** One exported artefact entry with digests and the full payload. */
export interface EvidenceCartExportEntry {
  classification: string;
  id: string;
  kind: EvidenceCartItemKind;
  label: string;
  /**
   * Identifier of the receipt the payload carries, or `null` when it carries
   * none. It names the originating run, so an export made in a later session
   * still points at the exact run rather than at whatever is current.
   */
  receipt_id: string | null;
  /**
   * Canonical JSON payload of the queued artefact (must round-trip with
   * ``payload_sha256``).
   */
  payload: unknown;
  /** SHA-256 hex digest of the canonical JSON payload. */
  payload_sha256: string;
  queued_at_utc: string;
  source_name: string | null;
}

/** Single export bundle for the whole cart. */
export interface EvidenceCartExportBundle {
  /** SHA-256 hex digest over the canonical form of entries + metadata. */
  bundle_sha256: string;
  entry_count: number;
  entries: EvidenceCartExportEntry[];
  /** ISO-8601 UTC export timestamp. */
  exported_at_utc: string;
  kind_counts: Record<string, number>;
  schema_version: typeof EVIDENCE_CART_SCHEMA_VERSION;
}

export type EvidenceCartEnqueueResult =
  | { ok: true; cart: EvidenceCart; item: EvidenceCartItem }
  | { ok: false; error: string; cart: EvidenceCart };

/**
 * Return an empty cart with the current schema version.
 */
export function emptyEvidenceCart(): EvidenceCart {
  return {
    items: [],
    schema_version: EVIDENCE_CART_SCHEMA_VERSION,
  };
}

/**
 * Enqueue a simulation, analysis, or other run artefact into the cart.
 *
 * Rejects empty labels and non-serialisable payloads. Does not mutate the input
 * cart; returns a new cart with the item appended.
 */
export function enqueueEvidenceCartArtefact(
  cart: EvidenceCart,
  draft: EvidenceCartArtefactDraft,
  options: { id?: string; nowUtc?: string } = {},
): EvidenceCartEnqueueResult {
  const label = draft.label.trim();
  if (label.length === 0) {
    return { ok: false, error: "Evidence cart label must not be empty", cart };
  }
  if (!isJsonSerialisable(draft.payload)) {
    return {
      ok: false,
      error: "Evidence cart payload must be JSON-serialisable",
      cart,
    };
  }
  const item: EvidenceCartItem = {
    classification: draft.classification.trim() || "unclassified",
    id: options.id ?? newEvidenceCartItemId(),
    kind: draft.kind,
    label,
    payload: draft.payload,
    queuedAtUtc: options.nowUtc ?? new Date().toISOString(),
    sourceName: draft.sourceName?.trim() || undefined,
  };
  return {
    ok: true,
    cart: {
      schema_version: EVIDENCE_CART_SCHEMA_VERSION,
      items: [...cart.items, item],
    },
    item,
  };
}

/**
 * Remove a cart item by id. Returns the same cart reference when the id is
 * missing (no silent create).
 */
export function removeEvidenceCartArtefact(
  cart: EvidenceCart,
  itemId: string,
): EvidenceCart {
  if (!cart.items.some((item) => item.id === itemId)) {
    return cart;
  }
  return {
    schema_version: EVIDENCE_CART_SCHEMA_VERSION,
    items: cart.items.filter((item) => item.id !== itemId),
  };
}

/**
 * Clear all items from the cart.
 */
export function clearEvidenceCart(): EvidenceCart {
  return emptyEvidenceCart();
}

/**
 * Build a single export bundle with per-item and bundle digests.
 *
 * Digests are computed from canonical JSON so the same payloads always yield
 * the same hex digest regardless of object key insertion order.
 */
export async function buildEvidenceCartExport(
  cart: EvidenceCart,
  options: { exportedAtUtc?: string } = {},
): Promise<EvidenceCartExportBundle | { error: string }> {
  if (cart.items.length === 0) {
    return { error: "Evidence cart is empty; queue at least one artefact" };
  }
  const exportedAtUtc = options.exportedAtUtc ?? new Date().toISOString();
  const entries: EvidenceCartExportEntry[] = [];
  const kindCounts: Record<string, number> = {};
  for (const item of cart.items) {
    const payloadSha = await sha256HexOfCanonicalJson(item.payload);
    kindCounts[item.kind] = (kindCounts[item.kind] ?? 0) + 1;
    entries.push({
      classification: item.classification,
      id: item.id,
      kind: item.kind,
      label: item.label,
      payload: item.payload,
      payload_sha256: payloadSha,
      queued_at_utc: item.queuedAtUtc,
      receipt_id: evidenceCartReceiptId(item.payload),
      source_name: item.sourceName ?? null,
    });
  }
  // Bundle digest excludes raw payloads so it remains stable metadata; each
  // entry still carries payload + payload_sha256 for full reconstruction.
  const digestBody = {
    entry_count: entries.length,
    entries: entries.map((entry) => ({
      classification: entry.classification,
      id: entry.id,
      kind: entry.kind,
      label: entry.label,
      payload_sha256: entry.payload_sha256,
      queued_at_utc: entry.queued_at_utc,
      receipt_id: entry.receipt_id,
      source_name: entry.source_name,
    })),
    exported_at_utc: exportedAtUtc,
    kind_counts: kindCounts,
    schema_version: EVIDENCE_CART_SCHEMA_VERSION,
  };
  const bundleSha = await sha256HexOfCanonicalJson(digestBody);
  return {
    entry_count: entries.length,
    entries,
    exported_at_utc: exportedAtUtc,
    kind_counts: kindCounts,
    schema_version: EVIDENCE_CART_SCHEMA_VERSION,
    bundle_sha256: bundleSha,
  };
}

/**
 * Verify that every export entry's ``payload_sha256`` matches its payload and
 * that the bundle digest matches the metadata-only digest body.
 */
export async function verifyEvidenceCartExportRoundTrip(
  bundle: EvidenceCartExportBundle,
): Promise<{ ok: true } | { ok: false; error: string }> {
  for (const entry of bundle.entries) {
    const actual = await sha256HexOfCanonicalJson(entry.payload);
    if (actual !== entry.payload_sha256) {
      return {
        ok: false,
        error: `payload_sha256 mismatch for entry ${entry.id}`,
      };
    }
  }
  const digestBody = {
    entry_count: bundle.entry_count,
    entries: bundle.entries.map((entry) => ({
      classification: entry.classification,
      id: entry.id,
      kind: entry.kind,
      label: entry.label,
      payload_sha256: entry.payload_sha256,
      queued_at_utc: entry.queued_at_utc,
      receipt_id: entry.receipt_id,
      source_name: entry.source_name,
    })),
    exported_at_utc: bundle.exported_at_utc,
    kind_counts: bundle.kind_counts,
    schema_version: bundle.schema_version,
  };
  const recomputed = await sha256HexOfCanonicalJson(digestBody);
  if (recomputed !== bundle.bundle_sha256) {
    return { ok: false, error: "bundle_sha256 mismatch after round-trip" };
  }
  return { ok: true };
}

/**
 * Serialise an export bundle to a downloadable JSON Blob.
 */
export function evidenceCartExportToBlob(bundle: EvidenceCartExportBundle): Blob {
  const text = `${JSON.stringify(bundle, null, 2)}\n`;
  return new Blob([text], { type: "application/json" });
}

/**
 * Suggested download filename for a cart export.
 */
export function evidenceCartExportFilename(bundle: EvidenceCartExportBundle): string {
  const short = bundle.bundle_sha256.slice(0, 12);
  return `studio-evidence-cart-${short}.json`;
}

/**
 * True when the cart has both a simulation and an analysis artefact (Phase 0
 * explore-one-model success path attachment).
 */
export function evidenceCartHasSimAndAnalysis(cart: EvidenceCart): boolean {
  const kinds = new Set(cart.items.map((item) => item.kind));
  return kinds.has("simulation") && kinds.has("analysis");
}

/**
 * Draft a simulation cart artefact from a model name and run result payload.
 */
export function simulationCartDraft(
  sourceName: string,
  payload: unknown,
  classification = "simulation",
): EvidenceCartArtefactDraft {
  return {
    classification,
    kind: "simulation",
    label: `Simulation: ${sourceName}`,
    payload,
    sourceName,
  };
}

/**
 * Draft an analysis cart artefact from a model name and analysis result payload.
 */
export function analysisCartDraft(
  sourceName: string,
  payload: unknown,
  classification = "analysis",
): EvidenceCartArtefactDraft {
  return {
    classification,
    kind: "analysis",
    label: `Analysis: ${sourceName}`,
    payload,
    sourceName,
  };
}

/**
 * Return the receipt identifier a queued payload carries, or `null`.
 */
export function evidenceCartReceiptId(payload: unknown): string | null {
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  const receipt = (payload as Record<string, unknown>).evidence_receipt;
  if (receipt === null || typeof receipt !== "object" || Array.isArray(receipt)) {
    return null;
  }
  const identifier = (receipt as Record<string, unknown>).receipt_id;
  return typeof identifier === "string" && identifier.length > 0 ? identifier : null;
}

/**
 * Compute SHA-256 hex of the shared canonical seal encoding of `value`.
 */
export async function sha256HexOfCanonicalJson(value: unknown): Promise<string> {
  return sealSha256(value);
}

/**
 * Canonical seal text: sorted object keys, no insignificant whitespace, stable
 * array order, one normal form per number, finite numbers only.
 */
export function canonicalJsonString(value: unknown): string {
  return canonicalSealText(value);
}

function isJsonSerialisable(value: unknown): boolean {
  try {
    canonicalJsonString(value);
    return true;
  } catch {
    return false;
  }
}

function newEvidenceCartItemId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `ec_${crypto.randomUUID()}`;
  }
  return `ec_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}
