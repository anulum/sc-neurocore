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
import EvidenceCartStrip from "./EvidenceCartStrip";

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
    expect(html).toContain("Last export:");
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
