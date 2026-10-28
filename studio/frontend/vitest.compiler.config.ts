// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Authored compiler frontend coverage

import { configDefaults, defineConfig } from "vitest/config";
import base from "./vitest.config.ts";

export default defineConfig({
  ...base,
  test: {
    ...base.test,
    include: [
      "src/equationIrConfig.test.ts", "src/compilerStoreState.test.ts",
      "src/components/EquationIrOptions.test.tsx", "src/components/CompilerInspector.test.tsx",
      "src/stores/studioEquationIrActions.test.ts", "src/stores/studioIR.test.ts",
      "src/stores/studioCompile.test.ts", "src/stores/studioBundle.test.ts",
      "src/studioBundleContext.test.ts", "src/studioProjectState.test.ts",
      "src/studioSavedSessions.test.ts",
      "e2e/compiler-inspector-live.test.tsx",
    ],
    exclude: configDefaults.exclude,
    environmentOptions: {
      happyDOM: {
        url: process.env.SC_NEUROCORE_STUDIO_COMPILER_API_ORIGIN ?? "http://127.0.0.1:24022",
        settings: {
          navigation: { disableMainFrameNavigation: true, disableFallbackToSetURL: true },
          fetch: { requestHeaders: [{
          url: /^http:\/\/127\.0\.0\.1:[0-9]+\/api(?:\/|$)/,
          headers: { "Sec-Fetch-Site": "same-origin" },
          }] },
        },
      },
    },
    maxWorkers: 1,
    fileParallelism: false,
    coverage: {
      enabled: true,
      provider: "v8",
      include: [
        "src/equationIrConfig.ts", "src/compilerStoreState.ts",
        "src/components/EquationIrOptions.tsx", "src/components/CompilerInspector.tsx",
        "src/stores/studioEquationIrActions.ts", "src/stores/studioCompile.ts",
        "src/api/compilerApi.ts", "src/studioBundleContext.ts",
        "src/studioProjectState.ts", "src/studioSavedSessions.ts",
      ],
      reporter: ["text", "json", "json-summary"],
      reportsDirectory: process.env.SC_NEUROCORE_STUDIO_COMPILER_COVERAGE_DIR ?? "coverage/compiler",
      thresholds: { perFile: true, statements: 100, branches: 100, functions: 100, lines: 100 },
    },
  },
});
