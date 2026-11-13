// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Evidence cart strip tests
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  emptyEvidenceCart,
  enqueueEvidenceCartArtefact,
  simulationCartDraft,
  type EvidenceCartExportBundle,
} from "../evidenceCart";
import EvidenceCartStrip, { type EvidenceCartStripProps } from "./EvidenceCartStrip";

/** One verified export of a one-entry cart. */
const verifiedExport: EvidenceCartExportBundle = {
  bundle_sha256: "ab".repeat(32),
  entry_count: 1,
  entries: [],
  exported_at_utc: "2026-07-19T12:00:00.000Z",
  kind_counts: { simulation: 1 },
  schema_version: "studio.evidence-cart.v1",
};

/**
 * Render the strip into a document and return the texts of its status regions, in order.
 *
 * @param props - The strip's props.
 * @returns Each status region's text.
 */
function statusTexts(props: EvidenceCartStripProps): string[] {
  const host = document.createElement("div");
  host.innerHTML = renderToStaticMarkup(<EvidenceCartStrip {...props} />);
  return [...host.querySelectorAll('[role="status"]')].map((region) => region.textContent);
}

/**
 * Return a cart holding one simulation artefact.
 *
 * @returns The cart.
 */
function oneArtefactCart(): EvidenceCartStripProps["cart"] {
  const queued = enqueueEvidenceCartArtefact(
    emptyEvidenceCart(),
    simulationCartDraft("AdEx", { spikes: [1] }),
    { id: "ec_1" },
  );
  if (!queued.ok) throw new Error(queued.error);
  return queued.cart;
}

describe("EvidenceCartStrip status messages", () => {
  it("keeps the count and the export outcome in statuses mounted before anything is queued", () => {
    expect(statusTexts({ cart: emptyEvidenceCart(), error: null, lastExport: null, onExport: () => undefined }))
      .toEqual(["0 artefacts", ""]);
  });

  it("announces the count with its kinds when an artefact is queued", () => {
    expect(statusTexts({ cart: oneArtefactCart(), error: null, lastExport: null, onExport: () => undefined })[0])
      .toBe("1 artefact (simulation:1)");
  });

  it("says in words that the last export passed its own round-trip verification", () => {
    expect(statusTexts({ cart: oneArtefactCart(), error: null, lastExport: verifiedExport, onExport: () => undefined })[1])
      .toBe(`Last export verified by its round trip: ${"ab".repeat(6)}… (1 entry)`);
  });

  it("counts a larger export's entries in the plural", () => {
    expect(statusTexts({
      cart: oneArtefactCart(), error: null, lastExport: { ...verifiedExport, entry_count: 3 }, onExport: () => undefined,
    })[1]).toBe(`Last export verified by its round trip: ${"ab".repeat(6)}… (3 entries)`);
  });

  it("announces a failed export in the same status", () => {
    expect(statusTexts({
      cart: oneArtefactCart(), error: "Evidence cart export failed.", lastExport: null, onExport: () => undefined,
    })[1]).toBe("Evidence cart export failed.");
  });
});

describe("EvidenceCartStrip", () => {
  it("renders queue counts and last export digest prefix", () => {
    const queued = enqueueEvidenceCartArtefact(
      emptyEvidenceCart(),
      simulationCartDraft("AdEx", { spikes: [1] }),
      { id: "ec_1" },
    );
    expect(queued.ok).toBe(true);
    if (!queued.ok) {
      return;
    }
    const lastExport: EvidenceCartExportBundle = {
      bundle_sha256: "ab".repeat(32),
      entry_count: 1,
      entries: [],
      exported_at_utc: "2026-07-19T12:00:00.000Z",
      kind_counts: { simulation: 1 },
      schema_version: "studio.evidence-cart.v1",
    };
    const html = renderToStaticMarkup(
      <EvidenceCartStrip
        cart={queued.cart}
        error={null}
        lastExport={lastExport}
        onExport={() => undefined}
      />,
    );
    expect(html).toContain("<strong>1</strong>");
    expect(html).toContain("artefact");
    expect(html).toContain("simulation:1");
    expect(html).toContain("Last export verified by its round trip:");
    expect(html).toContain("Export cart");
  });

  it("renders empty cart with disabled export affordance", () => {
    const html = renderToStaticMarkup(
      <EvidenceCartStrip
        cart={emptyEvidenceCart()}
        error={null}
        lastExport={null}
        onExport={() => undefined}
      />,
    );
    expect(html).toContain("<strong>0</strong>");
    expect(html).toContain("artefacts");
    expect(html).toContain("disabled=\"\"");
  });
});
