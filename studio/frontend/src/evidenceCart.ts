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

/** The schema every cart and every export declares. */
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

/** The new cart and the item queued, or the refusal and the cart unchanged. */
export type EvidenceCartEnqueueResult =
  | { ok: true; cart: EvidenceCart; item: EvidenceCartItem }
  | { ok: false; error: string; cart: EvidenceCart };

/**
 * Start an empty cart.
 *
 * @returns The cart, carrying the schema version its exports will declare.
 */
export function emptyEvidenceCart(): EvidenceCart {
  return {
    items: [],
    schema_version: EVIDENCE_CART_SCHEMA_VERSION,
  };
}

/**
 * Read a source name, treating a blank one as no name at all.
 *
 * This is a function rather than an inline `||` so the reason has somewhere to
 * live: a name of `"   "` is not a name, and `??` would keep it, putting an
 * empty label on a cart entry the reader has to identify later.
 *
 * @param sourceName - The name as it was typed, if one was.
 * @returns The trimmed name, or `undefined` when there is nothing left of it.
 */
function presentSourceName(sourceName: string | undefined): string | undefined {
  const trimmed = sourceName?.trim();
  return trimmed !== undefined && trimmed.length > 0 ? trimmed : undefined;
}

/**
 * Queue one run artefact.
 *
 * A blank label and a payload that will not survive JSON are both refused: the
 * cart is a ledger the reader comes back to, and an entry they cannot name or
 * that cannot be exported is worse than one they were told to fix.
 *
 * The cart is not mutated. A new one is returned with the item appended, so a
 * caller holding the old cart still holds what it held.
 *
 * @param cart - The cart as it stands.
 * @param draft - The artefact to queue.
 * @param options - An id and a timestamp, for reproducible tests; both are
 *   generated when absent.
 * @returns The new cart and the item, or the refusal with the cart unchanged.
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
    sourceName: presentSourceName(draft.sourceName),
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
 * Remove one item.
 *
 * An id that is not in the cart returns the same cart, by reference. Nothing
 * is created and nothing is logged: removing something that is not there is
 * not an error, and a caller comparing references can see that nothing moved.
 *
 * @param cart - The cart as it stands.
 * @param itemId - The item to remove.
 * @returns The new cart, or the same one when there was nothing to remove.
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
 * Empty the cart.
 *
 * @returns A fresh empty cart. The old one is untouched, so a caller that kept
 *   a reference to it still has what the reader queued.
 */
export function clearEvidenceCart(): EvidenceCart {
  return emptyEvidenceCart();
}

/**
 * Turn the whole cart into one export bundle.
 *
 * Every payload gets its own digest and the bundle gets one of its own. The
 * bundle digest is taken over the metadata only -- ids, labels, payload
 * digests, timestamps -- and deliberately not over the payloads themselves, so
 * two exports of the same cart agree on the bundle digest while each entry
 * still carries what is needed to reconstruct and check it.
 *
 * The digests come from the shared evidence seal rather than `JSON.stringify`,
 * which is what lets a digest taken here be compared with one the server took.
 *
 * @param cart - The cart to export.
 * @param options - An export timestamp, for reproducible tests.
 * @returns The bundle, or the reason there is none. An empty cart is refused:
 *   a bundle of nothing would still carry a digest and read as evidence.
 */
export async function buildEvidenceCartExport(
  cart: EvidenceCart,
  options: { exportedAtUtc?: string } = {},
): Promise<EvidenceCartExportBundle | { error: string }> {
  if (cart.items.length === 0) {
    return { error: "Evidence cart is empty; queue at least one artefact" };
  }
  const exportedAtUtc = options.exportedAtUtc ?? new Date().toISOString();
  // Capture every payload before yielding: hashing and exported bytes must
  // describe the same snapshot, even if a caller later mutates its source.
  const items = cart.items.map((item) => ({ ...item,
    payload: JSON.parse(canonicalSealText(item.payload)) as unknown,
  }));
  const entries: EvidenceCartExportEntry[] = [];
  const kindCounts: Record<string, number> = {};
  for (const item of items) {
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
 * Check a bundle against itself.
 *
 * Recomputes every payload digest and the bundle digest. This is what makes
 * the export checkable by whoever receives it, and it is the reason the
 * digest body is built the same way in both places rather than remembered.
 *
 * @param bundle - The bundle to check.
 * @returns Whether it holds together, and which entry failed when it does not.
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
 * Serialise a bundle for download.
 *
 * @param bundle - The bundle.
 * @returns The bytes, indented and newline-terminated so the file reads well
 *   in a terminal and diffs line by line.
 */
export function evidenceCartExportToBlob(bundle: EvidenceCartExportBundle): Blob {
  const text = `${JSON.stringify(bundle, null, 2)}\n`;
  return new Blob([text], { type: "application/json" });
}

/**
 * Name the file a bundle is saved as.
 *
 * The name carries the first twelve characters of the bundle digest, so two
 * exports of different carts cannot overwrite each other in a downloads
 * folder and the file can be matched to its bundle by eye.
 *
 * @param bundle - The bundle.
 * @returns The filename.
 */
export function evidenceCartExportFilename(bundle: EvidenceCartExportBundle): string {
  const short = bundle.bundle_sha256.slice(0, 12);
  return `studio-evidence-cart-${short}.json`;
}

/**
 * Whether the cart holds both halves of a complete story.
 *
 * A run and an analysis of that run is the pair the guided flow attaches; one
 * without the other is a partial record.
 *
 * @param cart - The cart.
 * @returns Whether it holds at least one of each.
 */
export function evidenceCartHasSimAndAnalysis(cart: EvidenceCart): boolean {
  const kinds = new Set(cart.items.map((item) => item.kind));
  return kinds.has("simulation") && kinds.has("analysis");
}

/**
 * Draft a cart entry for a simulation run.
 *
 * @param sourceName - The model the run was of.
 * @param payload - The run's result.
 * @param classification - The evidence class to record it under.
 * @returns The draft, ready to queue.
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
 * Draft a cart entry for an analysis.
 *
 * @param sourceName - The model the analysis was of.
 * @param payload - The analysis's result.
 * @param classification - The evidence class to record it under.
 * @returns The draft, ready to queue.
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
 * Read the receipt identifier out of a queued payload.
 *
 * The identifier names the run that produced the payload, which is what lets
 * an export made in a later session still point at that exact run rather than
 * at whatever the panel is showing now.
 *
 * @param payload - The queued payload, whatever shape it is.
 * @returns The identifier, or `null` when the payload carries none.
 */
export function evidenceCartReceiptId(payload: unknown): string | null {
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  if (!("evidence_receipt" in payload)) {
    return null;
  }
  const { evidence_receipt: receipt } = payload;
  if (receipt === null || typeof receipt !== "object" || Array.isArray(receipt)
    || !("receipt_id" in receipt)) {
    return null;
  }
  const { receipt_id: identifier } = receipt;
  return typeof identifier === "string" && identifier.length > 0 ? identifier : null;
}

/**
 * Digest a value the way the seal does.
 *
 * @param value - The value.
 * @returns Its SHA-256, as lowercase hexadecimal.
 */
export async function sha256HexOfCanonicalJson(value: unknown): Promise<string> {
  return sealSha256(value);
}

/**
 * Render a value the way the seal renders it: sorted keys, no insignificant
 * whitespace, stable array order, one normal form per number, and finite
 * numbers only.
 *
 * @param value - The value.
 * @returns Its canonical text.
 * @throws {Error} When the value cannot be rendered canonically -- a
 *   non-finite number, or something JSON has no form for.
 */
export function canonicalJsonString(value: unknown): string {
  return canonicalSealText(value);
}

/**
 * Whether a payload can be rendered canonically, and so digested and exported.
 *
 * @param value - The payload.
 * @returns Whether it survives the canonical rendering.
 */
function isJsonSerialisable(value: unknown): boolean {
  try {
    canonicalJsonString(value);
    return true;
  } catch {
    return false;
  }
}

/**
 * Mint an id for a cart item.
 *
 * @returns An identifier unique within one cart.
 */
function newEvidenceCartItemId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `ec_${crypto.randomUUID()}`;
  }
  return `ec_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}
