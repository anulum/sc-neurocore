// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Real candidate process lifecycle and frontend coverage

import { configDefaults, defineConfig } from "vitest/config";
import base from "./vitest.config.ts";

export default defineConfig({
  ...base,
  test: {
    ...base.test,
    include: ["src/candidateJobs.test.ts", "src/candidateWorkbench.test.ts", "src/candidateCatalogue.test.ts",
      "src/components/CandidateCatalogueStart.test.tsx",
      "src/components/CandidateJobHistory.test.tsx", "src/components/CandidatePanel.test.tsx",
      "e2e/candidate-jobs-live.test.tsx"],
    exclude: configDefaults.exclude,
    maxWorkers: 1,
    fileParallelism: false,
    coverage: {
      enabled: true,
      provider: "v8",
      include: ["src/candidateJobs.ts", "src/useCandidateJobs.ts", "src/api/candidatesApi.ts",
        "src/candidateCatalogue.ts", "src/components/CandidateCatalogueStart.tsx",
        "src/components/CandidateJobHistory.tsx", "src/components/CandidatePanel.tsx"],
      reporter: ["text", "json", "json-summary"],
      reportsDirectory: process.env.SC_NEUROCORE_STUDIO_CANDIDATE_COVERAGE_DIR ?? "coverage/candidates",
      thresholds: { perFile: true, statements: 100, branches: 100, functions: 100, lines: 100 },
    },
  },
});
