// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import SnnStudioPanel from "./SnnStudioPanel";

const container = document.getElementById("root");
if (container === null) {
  // The element is in `index.html`; its absence means the document that loaded
  // this bundle is not the Studio's own. Saying so beats React's message about
  // a null container.
  throw new Error("SC-NeuroCore Studio: no #root element to mount into.");
}

createRoot(container).render(
  <StrictMode>
    <SnnStudioPanel />
  </StrictMode>
);
