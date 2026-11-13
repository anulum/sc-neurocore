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
      "src/components/GraphJobsPanel.test.tsx",
    ],
    // The frontend coverage ledger (scripts/coverage-ledger.ts) reads this
    // json-summary. Coverage is off by default; `npm run test:coverage-ledger`
    // turns it on. Test files and the three test-support modules are not source.
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "src/**/*.test.{ts,tsx}",
        "src/**/*.d.ts",
        "src/candidateTestServer.ts",
        "src/equationIrTestSupport.ts",
        "src/plots/mockPlotContext.ts",
      ],
      reporter: ["text", "json-summary"],
      reportsDirectory: "coverage/ledger",
    },
  },
});
