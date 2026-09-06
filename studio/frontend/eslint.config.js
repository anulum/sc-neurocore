// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scoped strict TypeScript lint and docstring policy

/**
 * Every function this repository maintains carries a docblock, in every
 * language it is written in — and the check is done by the strictest standard
 * tool for that language, not by something hand-rolled here.
 *
 * Python already had this: ruff's `D` rules (pydocstyle, numpy convention) run
 * against the audited file list in `docs/docstring_policy.toml`, which grows
 * package by package until `D` can be promoted to the global select. That
 * design is deliberate and it is copied here rather than reinvented.
 *
 * TypeScript had no linter at all. It now has `typescript-eslint` at
 * **strictTypeChecked + stylisticTypeChecked** and `eslint-plugin-jsdoc` at its
 * TypeScript error preset, with `require-jsdoc` extended to every function-like
 * declaration. Measured when this was introduced, the frontend carried **719
 * undocumented declarations across 151 files**, so the same incremental scope
 * applies: `AUDITED` lists the files whose whole surface has been documented
 * and which are enforced completely; everything else is not yet linted, and
 * the number left is written down rather than hidden.
 *
 * A **new** file belongs in `AUDITED`. That is the point: new code is
 * documented and strictly typed by default instead of joining the backlog.
 */

import js from "@eslint/js";
import jsdoc from "eslint-plugin-jsdoc";
import tseslint from "typescript-eslint";

/**
 * Files whose entire surface has been audited, documented and made safe to
 * enforce. Sorted, so an addition is one reviewable line of diff.
 */
const AUDITED = [
  "e2e/admin-operator-audit-archive.spec.ts",
  "e2e/admin-operator-capabilities.spec.ts",
  "e2e/admin-operator-evidence-bundle.spec.ts",
  "e2e/admin-operator-project-evidence.spec.ts",
  "e2e/admin-operator-status.spec.ts",
  "e2e/admin-operator-synthesis.spec.ts",
  "e2e/adminOperatorHarness.ts",
  "e2e/analysis-job-host.spec.ts",
  "e2e/catalogue-to-silicon-live.spec.ts",
  "e2e/experiment-export-live.spec.ts",
  "e2e/guided-operator-run.spec.ts",
  "e2e/module-federation-host.spec.ts",
  "e2e/network-canvas-live.spec.ts",
  "playwright.export.config.ts",
  "playwright.graph.config.ts",
  "src/api/adminApi.ts",
  "src/api/analysisApi.ts",
  "src/api/benchmarksApi.ts",
  "src/api/client.ts",
  "src/api/compilerApi.ts",
  "src/api/dclsApi.ts",
  "src/api/graphApi.ts",
  "src/api/http.ts",
  "src/api/modelsApi.ts",
  "src/api/progressApi.ts",
  "src/api/projectApi.ts",
  "src/api/simulationApi.ts",
  "src/api/synthApi.test.ts",
  "src/api/synthApi.ts",
  "src/api/trainingApi.ts",
  "src/api/types.ts",
  "src/arrayAt.test.ts",
  "src/arrayAt.ts",
  "src/components/NetworkGraphTable.test.tsx",
  "src/components/NetworkGraphTable.tsx",
  "src/components/PopulationEditor.test.tsx",
  "src/components/PopulationEditor.tsx",
  "src/components/ProjectionEditor.test.tsx",
  "src/components/ProjectionEditor.tsx",
  "src/contrastAudit.test.ts",
  "src/contrastAudit.ts",
  "src/docstringGate.test.ts",
  "src/paletteContrast.test.ts",
  "src/paletteContrast.ts",
  "src/evidenceSeal.test.ts",
  "src/evidenceSeal.ts",
  "src/simulationRaw.test.ts",
  "src/simulationRaw.ts",
  "src/stores/studioStoreActions.test.ts",
  "src/studioGraphDuplicate.test.ts",
  "src/studioGraphDuplicate.ts",
  "src/studioGraphHistory.test.ts",
  "src/studioGraphHistory.ts",
  "src/studioGraphTable.test.ts",
  "src/studioGraphTable.ts",
  "src/studioGraphValidation.test.ts",
  "src/studioGraphValidation.ts",
  "src/studioNodeDeletion.test.ts",
  "src/studioPopulationEditor.test.ts",
  "src/studioPopulationEditor.ts",
  "src/studioProjectionEditor.test.ts",
  "src/studioProjectionEditor.ts",
];

/**
 * How much of the frontend is still outside the audited scope.
 *
 * Recorded so the remaining work is readable rather than implied, and so a
 * later sweep can state what it actually reduced.
 */
export const LEGACY_OUTSIDE_SCOPE = {
  /**
   * Undocumented declarations outside the audited scope, as **ESLint** counts
   * them — the tool that enforces the rule, run over the whole frontend with
   * `eslint.measure.js`.
   *
   * The first figure recorded here was 719, taken with a regex over source
   * lines. The owner directive of 2026-09-06 is explicit that a heuristic may
   * size a problem but may not become a ceiling, and this is why: re-taken
   * with the real tool the same surface reports **1297**. The heuristic
   * understated the debt by nearly half, and had it been used as a ratchet it
   * would have enforced the estimate rather than the surface.
   */
  declarations: 1297,
  /**
   * Files carrying at least one of those declarations.
   *
   * Recorded as 190 when the figure was first taken, which counted files with
   * *any* finding from the measurement config rather than files with the rule
   * this figure is about. Re-counted by the stated method it is **176**. The
   * declaration count did not move; only the file count was measuring
   * something other than what it said.
   */
  files: 176,
  measuredBy:
    "eslint 10.10.0 + eslint-plugin-jsdoc 64.3.6, " +
    "`npx eslint . --config eslint.measure.js -f json`, jsdoc/require-jsdoc",
  measuredOn: "2026-09-06",
  measuredOnSourceSha: "62583819a1cd9c160b9988f82f8f15b7cb9cfe53",
  nodeVersion: "v22.23.1",
  typescriptVersion: "5.8.3",
  /**
   * The strictness step that used to be recorded here, now taken.
   *
   * `noUncheckedIndexedAccess` stops an index lookup pretending it always
   * finds something. It reported 198 errors when first measured and 218 by the
   * time the audited scope had grown; it is now **on** in `tsconfig.json` and
   * reports zero. Every site was narrowed, given a stated fallback, or read
   * through `src/arrayAt.ts`, which fails with the index and the length rather
   * than asserting the element exists. No `!` was used and no rule was
   * relaxed, so this number is not coming back.
   */
  noUncheckedIndexedAccessErrors: 0,
  /**
   * The `e2e` suite is fully audited as of 2026-09-07; nothing remains here.
   *
   * Closing it removed 2 454 lines of duplication: the six admin-operator
   * specs each carried an identical 500-line copy of the same mocked Studio,
   * and the unused fixtures the strict profile reported were a symptom of six
   * copies each using a different subset. They share
   * `e2e/adminOperatorHarness.ts` now.
   */
  e2eOutsideScope: {},
};

export default tseslint.config(
  {
    ignores: ["dist/**", "node_modules/**", "test-results/**", "playwright-report/**", "@mf-types/**"],
  },
  {
    files: AUDITED,
    extends: [
      js.configs.recommended,
      ...tseslint.configs.strictTypeChecked,
      ...tseslint.configs.stylisticTypeChecked,
      jsdoc.configs["flat/recommended-typescript-error"],
    ],
    languageOptions: {
      parserOptions: {
        // Type-aware rules need a project that contains the file. `src` is
        // covered by tsconfig.json; the audited e2e specs and Playwright
        // configs live outside it and are covered by tsconfig.audited.json.
        project: ["./tsconfig.json", "./tsconfig.audited.json"],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Every function a reader of the module meets carries a docblock, and a
      // docblock that says nothing is not one.
      //
      // The contexts are stated because the rule's own default covers
      // functions and classes only. Without them `/** */` on an interface
      // satisfied `require-jsdoc` and nothing else looked at it — the gate
      // accepted an empty docblock, which is the padding this scope exists to
      // refuse. Verified against that exact candidate.
      "jsdoc/require-description": [
        "error",
        {
          contexts: [
            "ArrowFunctionExpression",
            "ClassDeclaration",
            "ClassExpression",
            "FunctionDeclaration",
            "FunctionExpression",
            "MethodDefinition",
            "TSEnumDeclaration",
            "TSInterfaceDeclaration",
            "TSTypeAliasDeclaration",
          ],
        },
      ],
      "jsdoc/require-jsdoc": [
        "error",
        {
          contexts: [
            // An exported arrow is part of the module's surface and is
            // documented like any other function. A closure inside a function
            // body is a detail of that function, whose own docblock covers it.
            "ExportNamedDeclaration > VariableDeclaration > VariableDeclarator > ArrowFunctionExpression",
            "TSInterfaceDeclaration",
            "TSTypeAliasDeclaration",
            "TSEnumDeclaration",
          ],
          publicOnly: false,
          require: {
            ArrowFunctionExpression: false,
            ClassDeclaration: true,
            ClassExpression: true,
            FunctionDeclaration: true,
            FunctionExpression: false,
            MethodDefinition: true,
          },
        },
      ],
      // The types are in the signature; repeating them in the docblock is a
      // second copy free to drift from the first.
      "jsdoc/require-param-type": "off",
      "jsdoc/require-returns-type": "off",
      // One blank line between the prose and the tag block, so a long
      // description does not run into its own parameter table.
      "jsdoc/tag-lines": ["error", "any", { startLines: 1 }],
      // A React component takes one destructured props object whose fields are
      // documented on the props interface, which `require-jsdoc` already
      // demands. Repeating each field as `@param root0.thing` would be a
      // second copy of that documentation, free to drift from the first.
      "jsdoc/check-param-names": ["error", { checkDestructured: false }],
      "jsdoc/require-param": [
        "error",
        { checkDestructured: false, checkDestructuredRoots: false },
      ],
      // The hazard this rule exists for is `${someObject}` rendering as
      // "[object Object]", or a nullish value rendering as "null". A number
      // has exactly one text form and no surprise, so it is admitted; every
      // other non-string still has to be converted deliberately.
      "@typescript-eslint/restrict-template-expressions": [
        "error",
        { allowNumber: true },
      ],
    },
  },
  {
    // A test's fixtures and cases are documented at the file and case level;
    // requiring a parameter table on every arrow callback inside a `describe`
    // would produce ceremony, not description.
    files: AUDITED.filter(
      (file) => file.includes(".test.") || file.startsWith("e2e/"),
    ),
    plugins: { jsdoc },
    rules: {
      "jsdoc/require-jsdoc": [
        "error",
        {
          publicOnly: false,
          require: {
            ArrowFunctionExpression: false,
            ClassDeclaration: true,
            FunctionDeclaration: true,
            FunctionExpression: false,
            MethodDefinition: false,
          },
        },
      ],
      "jsdoc/require-param": "off",
      "jsdoc/require-returns": "off",
      // React's `act` takes an async callback whether or not the body awaits
      // anything; the async form is its contract, not a style slip.
      "@typescript-eslint/require-await": "off",
    },
  },
);
