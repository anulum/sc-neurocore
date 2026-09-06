// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The Studio's colour contract, checked against the sources that declare it.
 *
 * `contrastAudit.ts` measures what a real browser paints, which is the only
 * honest check for the DOM — but it can only see text nodes. Two surfaces are
 * invisible to it: text drawn into a `<canvas>`, which is pixels by the time
 * the DOM sees it, and text written into an exported SVG, which never reaches
 * a page at all. Both are ordinary product text and both must clear WCAG 2.2
 * AA, so this module states which foreground is rendered on which ground and
 * checks the declarations directly. The arithmetic is `contrastAudit.ts`'s,
 * not a second copy of it.
 */

import {
  CONTRAST_AA_NORMAL,
  composite,
  contrastRatio,
  parseRgba,
  type Rgba,
} from "./contrastAudit";

/**
 * Translucent grounds painted behind canvas text, as CSS colour strings.
 *
 * Named rather than inlined because the text on them has to clear AA against
 * the *composited* colour, not against the tint. The inhibitory button carries
 * a lighter tint than its excitatory twin because `#ff5252` is far darker than
 * `#4fc3f7`, so equal alphas do not give equal contrast.
 */
export const CANVAS_TINTS = {
  errorStrip: "rgba(255, 82, 82, 0.1)",
  excitatoryButton: "rgba(79, 195, 247, 0.2)",
  excitatoryNode: "rgba(79, 195, 247, 0.15)",
  inhibitoryButton: "rgba(255, 82, 82, 0.12)",
  inhibitoryNode: "rgba(255, 82, 82, 0.15)",
  pipelineFailed: "rgba(255, 82, 82, 0.05)",
  pipelineSucceeded: "rgba(129, 199, 132, 0.05)",
} as const;

/** One foreground, the stack it is painted on, and what it must reach. */
export interface PaletteRequirement {
  /** How the pair is named in a failure report. */
  label: string;
  /** Foreground colour: a hex literal, an `rgb()`/`rgba()` string, or `var(--token)`. */
  foreground: string;
  /** Background stack, topmost first; the last layer must be opaque. */
  ground: readonly string[];
  /** Ratio the pair must reach. */
  threshold: number;
  /** Why this pair exists, so a reader can tell whether it still does. */
  why: string;
}

/** A requirement that is not met, with the number that failed it. */
export interface PaletteFinding {
  /** The requirement's label. */
  label: string;
  /** The measured ratio, rounded to two decimals. */
  ratio: number;
  /** The ratio it had to reach. */
  threshold: number;
  /** The resolved foreground, as `r,g,b`. */
  foreground: string;
  /** The resolved ground, as `r,g,b`. */
  ground: string;
}

/** A declared colour that could not be resolved to a colour at all. */
export class PaletteResolutionError extends Error {
  /**
   * Name the declaration that could not be read.
   *
   * @param label - The requirement's label.
   * @param value - The declaration that failed to resolve.
   */
  constructor(label: string, value: string) {
    super(`${label}: cannot resolve ${value} to a colour`);
    this.name = "PaletteResolutionError";
  }
}

const HEX = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/;
const CUSTOM_PROPERTY = /(--[a-z0-9-]+)\s*:\s*([^;{}]+);/gi;
const VAR_REFERENCE = /^var\(\s*(--[a-z0-9-]+)\s*\)$/;

/**
 * Read the custom properties declared in a stylesheet.
 *
 * @param css - The stylesheet's text.
 * @returns Property name to declared value, last declaration winning.
 */
export function readCssTokens(css: string): Map<string, string> {
  const tokens = new Map<string, string>();
  CUSTOM_PROPERTY.lastIndex = 0;
  let matched = CUSTOM_PROPERTY.exec(css);
  while (matched !== null) {
    tokens.set(matched[1], matched[2].trim());
    matched = CUSTOM_PROPERTY.exec(css);
  }
  return tokens;
}

/**
 * Resolve one declared colour to `Rgba`.
 *
 * Accepts `#rgb`, `#rrggbb`, `rgb()`/`rgba()`, and a single `var(--token)`
 * indirection resolved through the supplied token table.
 *
 * @param value - The declared colour.
 * @param tokens - Custom properties read from the stylesheet.
 * @returns The colour, or `null` when it is not one this can read.
 */
export function parseCssColour(
  value: string,
  tokens: ReadonlyMap<string, string>,
): Rgba | null {
  const trimmed = value.trim();
  const reference = VAR_REFERENCE.exec(trimmed);
  if (reference !== null) {
    const referenced = tokens.get(reference[1]);
    return referenced === undefined ? null : parseCssColour(referenced, tokens);
  }
  if (HEX.test(trimmed)) {
    const digits = trimmed.slice(1);
    const wide =
      digits.length === 3
        ? digits
            .split("")
            .map((digit) => digit + digit)
            .join("")
        : digits;
    return {
      a: 1,
      b: parseInt(wide.slice(4, 6), 16),
      g: parseInt(wide.slice(2, 4), 16),
      r: parseInt(wide.slice(0, 2), 16),
    };
  }
  return parseRgba(trimmed);
}

/**
 * Composite a background stack down to the opaque colour a reader sees.
 *
 * @param layers - The stack, topmost first; the last layer must be opaque.
 * @param tokens - Custom properties read from the stylesheet.
 * @returns The composited colour, or `null` when a layer cannot be read.
 */
export function resolveGround(
  layers: readonly string[],
  tokens: ReadonlyMap<string, string>,
): Rgba | null {
  const resolved: Rgba[] = [];
  for (const layer of layers) {
    const colour = parseCssColour(layer, tokens);
    if (colour === null) return null;
    resolved.push(colour);
  }
  if (resolved.length === 0) return null;
  let ground = resolved[resolved.length - 1];
  if (ground.a !== 1) return null;
  for (let index = resolved.length - 2; index >= 0; index--) {
    ground = composite(resolved[index], ground);
  }
  return ground;
}

/**
 * Every ground the grey ramp is rendered on, as background stacks.
 *
 * The plain surfaces come from the stylesheet's own `--bg-*` tokens; the
 * composited ones are the tints the Network Canvas paints, each over the base
 * the component actually places it on rather than over a conservative guess.
 * The accent tint behind the "+ Exc" button is the lightest of them and is
 * therefore the binding constraint on `--text-muted`.
 */
export const TEXT_GROUNDS: readonly (readonly string[])[] = [
  ["var(--bg-primary)"],
  ["var(--bg-secondary)"],
  ["var(--bg-tertiary)"],
  ["var(--bg-hover)"],
  ["var(--accent-dim)"],
  [CANVAS_TINTS.excitatoryNode, "var(--bg-primary)"],
  [CANVAS_TINTS.inhibitoryNode, "var(--bg-primary)"],
  [CANVAS_TINTS.excitatoryButton, "var(--bg-secondary)"],
  [CANVAS_TINTS.inhibitoryButton, "var(--bg-secondary)"],
  [CANVAS_TINTS.errorStrip, "var(--bg-primary)"],
  [CANVAS_TINTS.pipelineFailed, "var(--bg-primary)"],
  [CANVAS_TINTS.pipelineSucceeded, "var(--bg-primary)"],
];

/** The two plot grounds a canvas or exported SVG paints text on. */
export const PLOT_GROUNDS: readonly (readonly string[])[] = [
  ["#0a0e14"],
  ["#0d1117"],
];

/**
 * Requirements for the grey ramp: every text token on every ground it meets.
 *
 * @returns One requirement per token and ground.
 */
export function textTokenRequirements(): PaletteRequirement[] {
  const requirements: PaletteRequirement[] = [];
  for (const token of ["--text-primary", "--text-secondary", "--text-muted"]) {
    for (const ground of TEXT_GROUNDS) {
      requirements.push({
        foreground: `var(${token})`,
        ground,
        label: `${token} on ${ground.join(" over ")}`,
        threshold: CONTRAST_AA_NORMAL,
        why: "The Studio renders body text at 9–11px, so the large-text allowance never applies.",
      });
    }
  }
  requirements.push({
    foreground: "var(--bg-primary)",
    ground: ["var(--text-muted)"],
    label: "--bg-primary text on a --text-muted ground",
    threshold: CONTRAST_AA_NORMAL,
    why: "The tier-1 readiness badge fills its chip with the muted token and writes on it.",
  });
  return requirements;
}

/**
 * Requirements for one badge colour map, in both roles it is used in.
 *
 * A badge colour is the chip's text while the filter is off and the chip's
 * ground while it is on, so each value has to clear AA twice.
 *
 * @param mapName - The map's name, for the failure report.
 * @param colours - Tag to colour, exactly as the component declares it.
 * @returns Two requirements per entry.
 */
export function badgeRequirements(
  mapName: string,
  colours: Readonly<Record<string, string>>,
): PaletteRequirement[] {
  const requirements: PaletteRequirement[] = [];
  for (const [tag, colour] of Object.entries(colours)) {
    requirements.push({
      foreground: colour,
      ground: ["var(--bg-tertiary)"],
      label: `${mapName}.${tag} as unselected chip text`,
      threshold: CONTRAST_AA_NORMAL,
      why: "An unselected chip writes the tag in its own colour on --bg-tertiary.",
    });
    requirements.push({
      foreground: "var(--bg-primary)",
      ground: [colour],
      label: `${mapName}.${tag} as a selected chip ground`,
      threshold: CONTRAST_AA_NORMAL,
      why: "A selected chip fills with the tag colour and writes --bg-primary on it.",
    });
  }
  return requirements;
}

/**
 * Requirements for colours painted as text into a plot canvas or SVG export.
 *
 * @param colours - The colour literals, from the scan or from a constant.
 * @param origin - Where they came from, for the failure report.
 * @returns One requirement per colour and plot ground.
 */
export function plotTextRequirements(
  colours: readonly string[],
  origin: string,
): PaletteRequirement[] {
  const requirements: PaletteRequirement[] = [];
  for (const colour of colours) {
    for (const ground of PLOT_GROUNDS) {
      requirements.push({
        foreground: colour,
        ground,
        label: `${origin} paints ${colour} as text on ${ground.join(" over ")}`,
        threshold: CONTRAST_AA_NORMAL,
        why: "Canvas and SVG text is 9–11px and no DOM audit can measure it.",
      });
    }
  }
  return requirements;
}

/**
 * Check every requirement and return the ones that fail.
 *
 * @param requirements - The pairs to check.
 * @param tokens - Custom properties read from the stylesheet.
 * @returns The failures, in the order the requirements were given.
 * @throws {PaletteResolutionError} When a declared colour cannot be read; a
 *   pair that cannot be resolved is never reported as passing.
 */
export function auditPalette(
  requirements: readonly PaletteRequirement[],
  tokens: ReadonlyMap<string, string>,
): PaletteFinding[] {
  const findings: PaletteFinding[] = [];
  for (const requirement of requirements) {
    const foreground = parseCssColour(requirement.foreground, tokens);
    if (foreground === null) {
      throw new PaletteResolutionError(requirement.label, requirement.foreground);
    }
    const ground = resolveGround(requirement.ground, tokens);
    if (ground === null) {
      throw new PaletteResolutionError(requirement.label, requirement.ground.join(" over "));
    }
    const painted = foreground.a === 1 ? foreground : composite(foreground, ground);
    const ratio = contrastRatio(painted, ground);
    if (ratio + 1e-9 < requirement.threshold) {
      findings.push({
        foreground: `${String(Math.round(painted.r))},${String(Math.round(painted.g))},${String(Math.round(painted.b))}`,
        ground: `${String(Math.round(ground.r))},${String(Math.round(ground.g))},${String(Math.round(ground.b))}`,
        label: requirement.label,
        ratio: Math.round(ratio * 100) / 100,
        threshold: requirement.threshold,
      });
    }
  }
  return findings;
}

/** What a canvas source paints its text with. */
export interface CanvasTextColours {
  /** Colour literals assigned to `fillStyle` immediately before a `fillText`. */
  literals: string[];
  /** Identifiers assigned instead of a literal, which must be checked separately. */
  symbols: string[];
}

const FILL_STYLE = /ctx\.fillStyle\s*=\s*(?:"([^"]+)"|([A-Za-z_$][\w$]*))/;

/**
 * Find the colours a canvas source paints text with.
 *
 * Canvas state is sequential, so the colour of a `fillText` is whatever was
 * last assigned to `fillStyle` above it. Scanning for that is exact for
 * literals and names the leading identifier otherwise, so a colour introduced
 * through a variable or an indexed constant cannot slip past unnamed.
 *
 * @param source - The module's text.
 * @returns The literals and the identifiers, each deduplicated.
 * @throws {Error} When the source paints no text at all, which means the scan
 *   has stopped matching the code rather than that the code is clean.
 */
export function canvasTextColours(source: string): CanvasTextColours {
  const literals = new Set<string>();
  const symbols = new Set<string>();
  let current: { literal: string | null; symbol: string | null } | null = null;
  let painted = 0;
  for (const line of source.split("\n")) {
    const assigned = FILL_STYLE.exec(line);
    if (assigned !== null) {
      // Only one of the two alternation groups participates in any match, so
      // the other is `undefined` at runtime. The index type claims otherwise
      // only because `noUncheckedIndexedAccess` is not on yet; the casts state
      // the runtime truth rather than letting the checks below be read as dead.
      const literal = assigned[1] as string | undefined;
      const symbol = assigned[2] as string | undefined;
      current = {
        literal: literal ?? null,
        symbol: symbol ?? null,
      };
    }
    if (!line.includes("fillText(")) continue;
    painted += 1;
    if (current === null) continue;
    if (current.literal !== null) literals.add(current.literal);
    if (current.symbol !== null) symbols.add(current.symbol);
  }
  if (painted === 0) {
    throw new Error("no fillText call found; the scan no longer matches the source");
  }
  return { literals: [...literals], symbols: [...symbols] };
}
