// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model silicon capability strip

import { useEffect, useState } from "react";

import { fetchModelCapabilities, type ModelCapabilities } from "../api/client";
import { capabilityRows } from "../modelCapabilities";

/**
 * Which silicon operations this installation can run for the selected model.
 *
 * @param props - The selected model's name.
 * @returns The strip, or nothing until the matrix has arrived.
 */
export default function ModelCapabilitiesStrip({ modelName }: { modelName: string }) {
  const [capabilities, setCapabilities] = useState<ModelCapabilities | null>(null);

  useEffect(() => {
    let current = true;
    setCapabilities(null);
    fetchModelCapabilities(modelName)
      .then((result) => { if (current) setCapabilities(result); })
      .catch(() => { if (current) setCapabilities(null); });
    return () => { current = false; };
  }, [modelName]);

  if (capabilities === null) return null;
  return (
    <ul
      aria-label="Silicon operations for this model"
      data-testid="model-capabilities"
      style={{ listStyle: "none", padding: 0, margin: "4px 0", fontSize: 9 }}
    >
      {capabilityRows(capabilities).map((row) => (
        <li key={row.label} style={{ color: row.enabled ? "var(--text-secondary)" : "var(--text-muted)" }}>
          <span aria-hidden="true">{row.enabled ? "✓" : "✗"}</span>{" "}
          <strong>{row.label}</strong>
          <span>{row.enabled ? " — " : " — not available: "}{row.detail}</span>
        </li>
      ))}
    </ul>
  );
}
