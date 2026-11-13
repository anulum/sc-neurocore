// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Session evidence cart summary strip
import type { EvidenceCart, EvidenceCartExportBundle } from "../evidenceCart";

/** What is gathered for the next evidence bundle, and how to clear it. */
export interface EvidenceCartStripProps {
  cart: EvidenceCart;
  error: string | null;
  lastExport: EvidenceCartExportBundle | null;
  onExport: () => void;
}

/**
 * Compact left-panel strip for the session evidence cart.
 *
 * Shows the queue size, the last export's digest prefix, and the one action
 * that serialises the whole cart — simulation, analysis and other artefacts
 * together.
 *
 * The count and the export's outcome are two statuses, drawn even while empty,
 * so a screen reader announces an artefact being queued, an export and a
 * failure. An export is only offered after its own round-trip verification,
 * and the outcome says so in words.
 *
 * @param props - The cart's state and the export action.
 * @returns The strip.
 */
export default function EvidenceCartStrip({
  cart,
  error,
  lastExport,
  onExport,
}: EvidenceCartStripProps) {
  const kindSummary = cart.items
    .reduce<Record<string, number>>((counts, item) => {
      counts[item.kind] = (counts[item.kind] ?? 0) + 1;
      return counts;
    }, {});
  const kindText = Object.entries(kindSummary)
    .map(([kind, count]) => `${kind}:${count}`)
    .join(" · ");

  return (
    <div data-testid="evidence-cart-strip" style={{ fontSize: "var(--fs-body)", color: "var(--text-secondary)" }}>
      <div className="panel-header">Evidence cart</div>
      <div style={{ padding: "4px 6px", lineHeight: 1.45 }}>
        <div role="status">
          <strong>{cart.items.length}</strong> artefact{cart.items.length === 1 ? "" : "s"}
          {kindText ? ` (${kindText})` : ""}
        </div>
        <div role="status">
          {lastExport !== null && (
            <div title={lastExport.bundle_sha256}>
              Last export verified by its round trip: {lastExport.bundle_sha256.slice(0, 12)}…
              {` (${lastExport.entry_count} entr${lastExport.entry_count === 1 ? "y" : "ies"})`}
            </div>
          )}
          {error !== null && (
            <div style={{ color: "var(--danger, #f85149)" }}>{error}</div>
          )}
        </div>
        <button
          type="button"
          className="btn-simulate"
          disabled={cart.items.length === 0}
          onClick={onExport}
          style={{
            marginTop: 4,
            padding: "2px 7px",
            fontSize: "var(--fs-body)",
            opacity: cart.items.length === 0 ? 0.45 : 1,
          }}
        >
          Export cart
        </button>
      </div>
    </div>
  );
}
