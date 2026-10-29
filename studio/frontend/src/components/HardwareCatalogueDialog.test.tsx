// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hardware catalogue modal lifecycle contract
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import * as api from "../api/hardwareApi";
import HardwareCatalogueDialog from "./HardwareCatalogueDialog";

it("opens a native modal, isolates shortcuts and forwards its real close event", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.spyOn(api, "fetchHardwareCatalogue").mockResolvedValue({
    schema_version: "studio.hardware-catalogue.v1", entries: [],
    evidence_scope: "Registration only",
  });
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  const closed = vi.fn();
  const globalShortcut = vi.fn();
  window.addEventListener("keydown", globalShortcut);
  try {
    await act(async () => {
      root.render(<HardwareCatalogueDialog selectedTarget="ice40" modelMode={true}
        onSelectTarget={vi.fn()} onClose={closed} />);
    });
    const dialog = host.querySelector("dialog");
    if (dialog === null) throw new Error("Hardware dialog did not mount");
    expect(dialog.open).toBe(true);
    expect(host.querySelector("details")?.open).toBe(true);
    expect(host.textContent).toContain("0 matching registrations");
    await act(async () => {
      dialog.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true }));
    });
    expect(globalShortcut).not.toHaveBeenCalled();
    await act(async () => { dialog.close(); });
    expect(closed).toHaveBeenCalledOnce();
  } finally {
    await act(async () => { root.unmount(); });
    host.remove();
    window.removeEventListener("keydown", globalShortcut);
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  }
});
