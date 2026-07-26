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

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SnnStudioPanel />
  </StrictMode>
);
