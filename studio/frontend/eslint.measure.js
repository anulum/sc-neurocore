// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Measurement-only lint config for documentation debt

/**
 * The documentation rules applied to the WHOLE frontend, for measurement only.
 *
 * `eslint.config.js` enforces the audited scope and must stay green.
 * This one enforces nothing: it exists so the debt outside that scope is
 * counted by the tool that would enforce it rather than by a heuristic, which
 * the owner directive of 2026-09-06 forbids as a ceiling. Driven by
 * `tools/documentation_debt.py`; it is deliberately not wired into
 * `npm run lint`.
 *
 * Type-aware rules are absent here on purpose: the whole tree is not in a
 * TypeScript project, and this figure is about documentation, not types.
 */

// Measurement-only config: the documentation rules applied to the WHOLE
// frontend, so the figure recorded as debt is taken by the tool that enforces
// it rather than by a heuristic.
import jsdoc from "eslint-plugin-jsdoc";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", "test-results/**", "playwright-report/**", "@mf-types/**"] },
  {
    files: ["src/**/*.ts", "src/**/*.tsx", "e2e/**/*.ts", "playwright*.config.ts", "vite.config.ts"],
    extends: [jsdoc.configs["flat/recommended-typescript-flavor-error"]],
    languageOptions: { parser: tseslint.parser, parserOptions: { ecmaFeatures: { jsx: true } } },
    rules: {
      "jsdoc/require-jsdoc": ["error", {
        contexts: [
          "ExportNamedDeclaration > VariableDeclaration > VariableDeclarator > ArrowFunctionExpression",
          "TSInterfaceDeclaration", "TSTypeAliasDeclaration", "TSEnumDeclaration",
        ],
        publicOnly: false,
        require: { ArrowFunctionExpression: false, ClassDeclaration: true, ClassExpression: true,
                   FunctionDeclaration: true, FunctionExpression: false, MethodDefinition: true },
      }],
      "jsdoc/require-description": "error",
      "jsdoc/require-param": "off",
      "jsdoc/require-returns": "off",
      "jsdoc/require-param-type": "off",
      "jsdoc/require-returns-type": "off",
    },
  },
);
