// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — real Module Federation host harness

import { loadRemote, registerRemotes } from "@module-federation/runtime";
import type { ComponentType } from "react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

interface FederatedPanelModule {
  default: ComponentType;
}

const container = document.getElementById("root");
if (container === null) {
  throw new Error("Federation host requires a #root element.");
}

registerRemotes([
  {
    entry: "http://127.0.0.1:5184/studios/sc-neurocore/remoteEntry.js",
    name: "sc_neurocore",
    type: "module",
  },
]);

const remote = await loadRemote<FederatedPanelModule>("sc_neurocore/SnnStudioPanel");
if (remote === null || typeof remote.default !== "function") {
  throw new Error("The sc_neurocore remote did not expose SnnStudioPanel.");
}

const SnnStudioPanel = remote.default;
container.dataset.federationStatus = "loaded";
createRoot(container).render(
  <StrictMode>
    <SnnStudioPanel />
  </StrictMode>,
);
