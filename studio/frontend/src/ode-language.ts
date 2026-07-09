// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Monaco language definition for neuron ODE equations

import type { languages } from "monaco-editor";

export const ODE_LANGUAGE_ID = "sc-ode";

export const odeLanguageConfig: languages.LanguageConfiguration = {
  comments: { lineComment: "#" },
  brackets: [["(", ")"]],
  autoClosingPairs: [
    { open: "(", close: ")" },
  ],
};

export const odeTokensProvider: languages.IMonarchLanguage = {
  defaultToken: "",
  tokenPostfix: ".ode",

  keywords: [
    "threshold", "reset", "unless", "refractory",
  ],

  operators: ["+", "-", "*", "/", "=", ">", "<", ">=", "<="],

  // State variable patterns: dv/dt, dw/dt, dm/dt etc.
  // Parameter patterns: E_L, tau_m, C_m, g_Na etc.
  tokenizer: {
    root: [
      // Comments
      [/#.*$/, "comment"],

      // ODE derivative: dv/dt, dw/dt, dm/dt
      [/d[a-zA-Z_]\w*\/dt/, "keyword.derivative"],

      // Threshold/reset directives
      [/\b(threshold|reset|unless|refractory)\b/, "keyword"],

      // Numbers (float and int)
      [/-?\d+\.\d*([eE][-+]?\d+)?/, "number.float"],
      [/-?\d+([eE][-+]?\d+)?/, "number"],

      // Common neuron parameters (uppercase or with underscore prefix)
      [/\b(E_[A-Za-z]+|V_[A-Za-z]+|g_[A-Za-z]+|tau_[A-Za-z]+|C_[A-Za-z]+|I_[A-Za-z]*)\b/, "variable.parameter"],

      // Built-in functions
      [/\b(exp|log|sqrt|abs|sin|cos|tanh|sigmoid|heaviside|min|max)\b/, "support.function"],

      // Common variables (single letter or short)
      [/\b(v|w|u|m|h|n|s|r|q|I|t|dt)\b/, "variable.state"],

      // Generic identifiers
      [/[a-zA-Z_]\w*/, "identifier"],

      // Operators
      [/[+\-*/=><]/, "operator"],

      // Parentheses
      [/[()]/, "delimiter.parenthesis"],

      // Whitespace
      [/\s+/, "white"],
    ],
  },
};

export const odeThemeRules: { token: string; foreground: string; fontStyle?: string }[] = [
  { token: "comment.ode", foreground: "6A9955" },
  { token: "keyword.derivative.ode", foreground: "569CD6", fontStyle: "bold" },
  { token: "keyword.ode", foreground: "C586C0" },
  { token: "number.float.ode", foreground: "B5CEA8" },
  { token: "number.ode", foreground: "B5CEA8" },
  { token: "variable.parameter.ode", foreground: "4EC9B0" },
  { token: "variable.state.ode", foreground: "9CDCFE" },
  { token: "support.function.ode", foreground: "DCDCAA" },
  { token: "operator.ode", foreground: "D4D4D4" },
  { token: "identifier.ode", foreground: "D4D4D4" },
];

export function registerODELanguage(monaco: typeof import("monaco-editor")) {
  if (monaco.languages.getLanguages().some((l) => l.id === ODE_LANGUAGE_ID)) return;

  monaco.languages.register({ id: ODE_LANGUAGE_ID });
  monaco.languages.setLanguageConfiguration(ODE_LANGUAGE_ID, odeLanguageConfig);
  monaco.languages.setMonarchTokensProvider(ODE_LANGUAGE_ID, odeTokensProvider);

  monaco.editor.defineTheme("sc-ode-dark", {
    base: "vs-dark",
    inherit: true,
    rules: odeThemeRules,
    colors: {
      "editor.background": "#0d1117",
      "editor.foreground": "#c9d1d9",
    },
  });
}
