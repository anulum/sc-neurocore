// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Session evidence cart summary strip
import type { EvidenceCart, EvidenceCartExportBundle } from "../evidenceCart";

export interface EvidenceCartStripProps {
  cart: EvidenceCart;
  error: string | null;
  lastExport: EvidenceCartExportBundle | null;
  onExport: () => void;
}

/**
 * Compact left-panel strip for the session evidence cart.
 *
 * Surfaces queue size, last export digest prefix, and the single-export action
 * that serialises the whole cart (simulation + analysis + other artefacts).
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
    <div data-testid="evidence-cart-strip" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
      <div className="panel-header">Evidence cart</div>
      <div style={{ padding: "4px 6px", lineHeight: 1.45 }}>
        <div>
          <strong>{cart.items.length}</strong> artefact{cart.items.length === 1 ? "" : "s"}
          {kindText ? ` (${kindText})` : ""}
        </div>
        {lastExport !== null && (
          <div title={lastExport.bundle_sha256}>
            Last export: {lastExport.bundle_sha256.slice(0, 12)}… ({lastExport.entry_count} entries)
          </div>
        )}
        {error !== null && (
          <div style={{ color: "var(--danger, #f85149)" }}>{error}</div>
        )}
        <button
          type="button"
          className="btn-simulate"
          disabled={cart.items.length === 0}
          onClick={onExport}
          style={{
            marginTop: 4,
            padding: "2px 7px",
            fontSize: 10,
            opacity: cart.items.length === 0 ? 0.45 : 1,
          }}
        >
          Export cart
        </button>
      </div>
    </div>
  );
}
