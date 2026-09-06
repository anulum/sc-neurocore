// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The colour contract, checked against the files that declare it.
 *
 * These cases read the real stylesheet and the real component constants, so a
 * colour change that breaks WCAG 2.2 AA fails here rather than in a browser
 * nobody ran. Everything the live audit cannot see — canvas text, exported SVG
 * text — is covered only here.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { relativeLuminance } from "./contrastAudit";

import {
  BEHAVIOR_COLORS,
  MATURITY_COLORS,
  PATTERN_COLORS,
} from "./components/ModelBrowser";
import {
  CANVAS_TINTS,
  PaletteResolutionError,
  auditPalette,
  badgeRequirements,
  canvasTextColours,
  parseCssColour,
  plotTextRequirements,
  readCssTokens,
  resolveGround,
  textTokenRequirements,
} from "./paletteContrast";
import { PLOT_AXIS, PLOT_COLORS } from "./simulationPlotCanvas";

const HERE = dirname(fileURLToPath(import.meta.url));

/**
 * Read one source file from this directory.
 *
 * @param relative - Path relative to `src/`.
 * @returns The file's text.
 */
function source(relative: string): string {
  return readFileSync(join(HERE, relative), "utf8");
}

const TOKENS = readCssTokens(source("index.css"));

describe("readCssTokens", () => {
  it("reads the declared custom properties", () => {
    expect(TOKENS.get("--text-muted")).toBe("#9ca4ae");
    expect(TOKENS.get("--bg-primary")).toBe("#0d1117");
  });

  it("keeps the last declaration when a property is declared twice", () => {
    const tokens = readCssTokens(":root { --x: #000000; }\n.dark { --x: #ffffff; }");
    expect(tokens.get("--x")).toBe("#ffffff");
  });

  it("returns nothing for a stylesheet with no custom properties", () => {
    expect(readCssTokens("body { color: red; }").size).toBe(0);
  });
});

describe("parseCssColour", () => {
  it("reads a six-digit hex", () => {
    expect(parseCssColour("#0d1117", TOKENS)).toEqual({ a: 1, b: 23, g: 17, r: 13 });
  });

  it("expands a three-digit hex", () => {
    expect(parseCssColour("#abc", TOKENS)).toEqual({ a: 1, b: 204, g: 187, r: 170 });
  });

  it("reads an rgba string", () => {
    expect(parseCssColour("rgba(255, 82, 82, 0.12)", TOKENS)).toEqual({
      a: 0.12,
      b: 82,
      g: 82,
      r: 255,
    });
  });

  it("follows a var() reference into the token table", () => {
    expect(parseCssColour("var(--bg-primary)", TOKENS)).toEqual({ a: 1, b: 23, g: 17, r: 13 });
  });

  it("returns null for a token that is not declared", () => {
    expect(parseCssColour("var(--not-a-token)", TOKENS)).toBeNull();
  });

  it("returns null for a colour it cannot read rather than guessing", () => {
    expect(parseCssColour("linear-gradient(red, blue)", TOKENS)).toBeNull();
    expect(parseCssColour("#12345", TOKENS)).toBeNull();
  });
});

describe("resolveGround", () => {
  it("composites a tint over its base", () => {
    const ground = resolveGround([CANVAS_TINTS.inhibitoryButton, "var(--bg-secondary)"], TOKENS);
    if (ground === null) throw new Error("the inhibitory button ground did not resolve");
    expect(Math.round(ground.r)).toBe(50);
  });

  it("returns an opaque single layer unchanged", () => {
    expect(resolveGround(["#0d1117"], TOKENS)).toEqual({ a: 1, b: 23, g: 17, r: 13 });
  });

  it("refuses a stack whose last layer is translucent", () => {
    expect(resolveGround(["rgba(0, 0, 0, 0.5)"], TOKENS)).toBeNull();
  });

  it("refuses an empty stack", () => {
    expect(resolveGround([], TOKENS)).toBeNull();
  });

  it("refuses a stack containing a layer it cannot read", () => {
    expect(resolveGround(["var(--nope)", "#000000"], TOKENS)).toBeNull();
  });
});

describe("auditPalette", () => {
  it("reports the ratio that failed", () => {
    const findings = auditPalette(
      [
        {
          foreground: "#484f58",
          ground: ["#0d1117"],
          label: "the ramp before it was raised",
          threshold: 4.5,
          why: "the defect this module exists for",
        },
      ],
      TOKENS,
    );
    expect(findings).toHaveLength(1);
    expect(findings[0]?.ratio).toBeCloseTo(2.28, 2);
    expect(findings[0]?.foreground).toBe("72,79,88");
  });

  it("composites a translucent foreground onto its ground before judging", () => {
    const findings = auditPalette(
      [
        {
          foreground: "rgba(255, 255, 255, 0.05)",
          ground: ["#0d1117"],
          label: "a foreground that is nearly transparent",
          threshold: 4.5,
          why: "a translucent foreground is not its opaque colour",
        },
      ],
      TOKENS,
    );
    expect(findings).toHaveLength(1);
    expect(findings[0]?.ratio).toBeLessThan(1.6);
  });

  it("throws rather than passing a pair it cannot resolve", () => {
    expect(() =>
      auditPalette(
        [
          {
            foreground: "var(--not-declared)",
            ground: ["#000000"],
            label: "an undeclared token",
            threshold: 4.5,
            why: "an unresolvable pair must never be silently green",
          },
        ],
        TOKENS,
      ),
    ).toThrow(PaletteResolutionError);
  });
});

describe("canvasTextColours", () => {
  it("attributes a fillText to the fillStyle above it", () => {
    const found = canvasTextColours('ctx.fillStyle = "#abcdef";\nctx.fillText("x", 0, 0);\n');
    expect(found.literals).toEqual(["#abcdef"]);
    expect(found.symbols).toEqual([]);
  });

  it("names an identifier instead of dropping it", () => {
    const found = canvasTextColours("ctx.fillStyle = COLORS[idx];\nctx.fillText(\"x\", 0, 0);\n");
    expect(found.symbols).toEqual(["COLORS"]);
    expect(found.literals).toEqual([]);
  });

  it("ignores a fill that paints an area rather than text", () => {
    const found = canvasTextColours(
      'ctx.fillStyle = "rgba(0,0,0,0.6)";\nctx.fillRect(0, 0, 1, 1);\n' +
        'ctx.fillStyle = "#abcdef";\nctx.fillText("x", 0, 0);\n',
    );
    expect(found.literals).toEqual(["#abcdef"]);
  });

  it("refuses a source that paints no text, rather than reporting it clean", () => {
    expect(() => canvasTextColours('ctx.fillStyle = "#abcdef";')).toThrow(
      /no fillText call found/,
    );
  });
});

describe("the Studio's declared colours", () => {
  it("holds the grey ramp to AA on every ground it is rendered on", () => {
    expect(auditPalette(textTokenRequirements(), TOKENS)).toEqual([]);
  });

  it("keeps the ramp's three steps ordered and distinct", () => {
    const luminance = (token: string): number => {
      const colour = parseCssColour(`var(${token})`, TOKENS);
      if (colour === null) throw new Error(`${token} is not declared`);
      return relativeLuminance(colour);
    };
    expect(luminance("--text-muted")).toBeLessThan(luminance("--text-secondary"));
    expect(luminance("--text-secondary")).toBeLessThan(luminance("--text-primary"));
  });

  it.each([
    ["MATURITY_COLORS", MATURITY_COLORS],
    ["PATTERN_COLORS", PATTERN_COLORS],
    ["BEHAVIOR_COLORS", BEHAVIOR_COLORS],
  ])("holds %s to AA in both of its roles", (name, colours) => {
    expect(auditPalette(badgeRequirements(name, colours), TOKENS)).toEqual([]);
  });

  it("holds every colour the plot canvas paints text with to AA", () => {
    const scanned = canvasTextColours(source("components/SimulationPlot.tsx"));
    expect(scanned.literals.length).toBeGreaterThan(5);
    expect(scanned.symbols.sort()).toEqual(["AXIS", "COLORS"]);
    const requirements = [
      ...plotTextRequirements(scanned.literals, "SimulationPlot"),
      ...plotTextRequirements([PLOT_AXIS], "PLOT_AXIS"),
      ...plotTextRequirements([...PLOT_COLORS], "PLOT_COLORS"),
    ];
    expect(auditPalette(requirements, TOKENS)).toEqual([]);
  });

  it("holds every colour the SVG export writes text with to AA", () => {
    const exporter = source("simulationExports.ts");
    const fills = [...exporter.matchAll(/<text\b[^>]*fill="(\$\{[A-Z_]+\}|#[0-9a-f]{6})"/g)].map(
      (match) => match[1],
    );
    expect(fills.length).toBeGreaterThan(3);
    const resolved = fills.map((fill) => (fill === "${PLOT_AXIS}" ? PLOT_AXIS : fill));
    expect(resolved.every((fill) => fill.startsWith("#"))).toBe(true);
    expect(auditPalette(plotTextRequirements(resolved, "simulationExports"), TOKENS)).toEqual([]);
  });

  it("keeps the canvas tints exactly as the components use them", () => {
    const canvas = source("components/NetworkCanvas.tsx");
    for (const name of Object.keys(CANVAS_TINTS)) {
      expect(canvas).toContain(`CANVAS_TINTS.${name}`);
    }
    expect(canvas).not.toMatch(/background:[^,;]*rgba\(/);
  });
});
