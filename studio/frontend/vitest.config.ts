// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // These HTTP cases use a real Python server and run in the native candidate
    // cohort, whose job installs the Studio backend before starting Vitest.
    exclude: [
      ...configDefaults.exclude,
      "e2e/**",
      "src/candidateCatalogue.test.ts",
      "src/components/CandidateCatalogueStart.test.tsx",
      "src/components/CandidatePanel.test.tsx",
    ],
  },
});
