// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Measuring text contrast, and refusing to score what it cannot resolve

/**
 * Colour contrast, computed rather than eyeballed.
 *
 * The live browser contract checks the properties an assistive technology
 * consumes — accessible names, focus, tab order — and has said in writing that
 * it does **not** cover contrast, because measuring it needs the resolved
 * colours of every text node against its effective background and a check that
 * looks at the wrong pixels passes for the wrong reason.
 *
 * This is that measurement, done properly:
 *
 * a background is **composited**, not read from one element. `rgba(0,0,0,0)`
 *   is the default, and most elements have it: the colour a reader actually
 *   sees is the first opaque ancestor with every translucent layer painted
 *   over it in order.
 * an element whose background **cannot** be resolved — a gradient, an image,
 *   or an ancestor chain that never reaches an opaque colour — is reported as
 *   `unresolved`, never scored. Assuming white would manufacture a pass for
 *   dark text on an unknown ground.
 * the threshold follows WCAG 2.2: 4.5:1 for body text, 3:1 for large text,
 *   which is 18.66px and up when bold, 24px and up otherwise.
 *
 * The formulae are the specification's own: sRGB channels linearised, relative
 * luminance weighted 0.2126/0.7152/0.0722, ratio `(L1+0.05)/(L2+0.05)`.
 */

/** One colour, with the alpha the browser reported. */
export interface Rgba {
  r: number;
  g: number;
  b: number;
  a: number;
}

/** The verdict for one text-bearing element. */
export interface ContrastSample {
  /** How the element is identified in a report. */
  label: string;
  /** The text whose contrast was measured, trimmed for the report. */
  text: string;
  /** Foreground as reported by the browser. */
  foreground: Rgba;
  /** Composited background, or `null` when it could not be resolved. */
  background: Rgba | null;
  /** Font size in CSS pixels. */
  fontSize: number;
  /** Whether the browser reports a weight of 700 or more. */
  bold: boolean;
}

/** What a measured sample is worth. */
export type ContrastVerdict = "pass" | "fail" | "unresolved";

/**
 * One sample, judged.
 *
 * A union rather than a nullable ratio, so a measured verdict always carries
 * its number: the alternative is a `number | null` that every reader has to
 * assert away, and an assertion is a claim the type does not support.
 */
export type ContrastResult =
  | {
      sample: ContrastSample;
      /** The measured ratio; always present when a verdict was reached. */
      ratio: number;
      /** The threshold this sample had to meet. */
      required: number;
      verdict: "pass" | "fail";
    }
  | {
      sample: ContrastSample;
      /** No ratio exists: the background could not be resolved. */
      ratio: null;
      required: number;
      verdict: "unresolved";
    };

/** WCAG 2.2 AA thresholds. */
export const CONTRAST_AA_NORMAL = 4.5;
/** The relaxed threshold large text is allowed. */
export const CONTRAST_AA_LARGE = 3;

/**
 * Parse a CSS colour of the form the browser reports.
 *
 * `getComputedStyle` normalises to `rgb(...)` or `rgba(...)`, so only those two
 * are accepted; anything else — a gradient, a keyword this build does not
 * expect — returns `null` rather than a guess.
 *
 * @param value - The computed colour string.
 * @returns The colour, or `null` when it is not one this can read.
 */
export function parseRgba(value: string): Rgba | null {
  const matched = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)(?:[\s,/]+([\d.]+))?\s*\)$/.exec(
    value.trim(),
  );
  if (matched === null) return null;
  // The alpha group is optional, so it is `undefined` at runtime whenever the
  // colour was written without one. The index type claims otherwise only
  // because `noUncheckedIndexedAccess` is not on yet, and the cast states the
  // runtime truth rather than letting the check be optimised away as dead.
  const alpha = matched[4] as string | undefined;
  return {
    a: alpha === undefined ? 1 : Number(alpha),
    b: Number(matched[3]),
    g: Number(matched[2]),
    r: Number(matched[1]),
  };
}

/**
 * Paint one translucent colour over another.
 *
 * Straight source-over compositing, which is what a browser does to a
 * background stack.
 *
 * @param over - The colour painted on top.
 * @param under - The colour beneath it, which must be opaque.
 * @returns The opaque result.
 */
export function composite(over: Rgba, under: Rgba): Rgba {
  const alpha = over.a;
  return {
    a: 1,
    b: over.b * alpha + under.b * (1 - alpha),
    g: over.g * alpha + under.g * (1 - alpha),
    r: over.r * alpha + under.r * (1 - alpha),
  };
}

/**
 * Return the background a reader actually sees behind an element.
 *
 * @param layers - Background colours from the element outwards, in paint
 *   order: the element's own first, then each ancestor's.
 * @returns The composited opaque colour, or `null` when the stack never
 *   reaches one — in which case nothing is known and nothing is claimed.
 */
export function resolveBackground(layers: readonly (Rgba | null)[]): Rgba | null {
  const stack: Rgba[] = [];
  for (const layer of layers) {
    // An unreadable layer — a gradient, an image — hides everything behind it,
    // so the stack cannot be resolved at all.
    if (layer === null) return null;
    if (layer.a === 0) continue;
    if (layer.a === 1) {
      // The opaque layer is the ground; everything already collected is
      // painted back over it, innermost last.
      return stack.reduceRight((under, over) => composite(over, under), layer);
    }
    stack.push(layer);
  }
  return null;
}

/**
 * Return the relative luminance of an opaque colour, per WCAG 2.2.
 *
 * @param colour - An opaque sRGB colour.
 * @returns Its relative luminance in [0, 1].
 */
export function relativeLuminance(colour: Rgba): number {
  const channel = (value: number): number => {
    const scaled = value / 255;
    return scaled <= 0.04045 ? scaled / 12.92 : ((scaled + 0.055) / 1.055) ** 2.4;
  };
  return (
    0.2126 * channel(colour.r) + 0.7152 * channel(colour.g) + 0.0722 * channel(colour.b)
  );
}

/**
 * Return the contrast ratio between two opaque colours.
 *
 * @param first - One colour.
 * @param second - The other.
 * @returns The ratio, from 1 (identical) to 21 (black on white).
 */
export function contrastRatio(first: Rgba, second: Rgba): number {
  const a = relativeLuminance(first);
  const b = relativeLuminance(second);
  const lighter = Math.max(a, b);
  const darker = Math.min(a, b);
  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Return the threshold one sample has to meet.
 *
 * Large text is 18.66px and up when bold, 24px and up otherwise — the
 * specification's own boundary, in CSS pixels.
 *
 * @param sample - The measured element.
 * @returns 3 for large text, 4.5 otherwise.
 */
export function requiredRatio(sample: Pick<ContrastSample, "fontSize" | "bold">): number {
  const large = sample.bold ? sample.fontSize >= 18.66 : sample.fontSize >= 24;
  return large ? CONTRAST_AA_LARGE : CONTRAST_AA_NORMAL;
}

/**
 * Judge one sample.
 *
 * @param sample - The measured element, with its composited background or
 *   `null` when that could not be resolved.
 * @returns The ratio, the threshold and the verdict; an unresolved background
 *   yields `unresolved` and no ratio, never a pass.
 */
export function judge(sample: ContrastSample): ContrastResult {
  const required = requiredRatio(sample);
  if (sample.background === null) {
    return { ratio: null, required, sample, verdict: "unresolved" };
  }
  const foreground =
    sample.foreground.a === 1
      ? sample.foreground
      : composite(sample.foreground, sample.background);
  const ratio = contrastRatio(foreground, sample.background);
  // Rounded to two places before comparing, so a value the report prints as
  // 4.5 is not failed by a difference nobody can see in it.
  const rounded = Math.round(ratio * 100) / 100;
  return { ratio: rounded, required, sample, verdict: rounded >= required ? "pass" : "fail" };
}

/**
 * Return one line per sample that did not pass.
 *
 * Failures and unresolved backgrounds are reported separately, because they
 * mean different things: one is a contrast the palette does not meet, the
 * other is a measurement that could not be taken.
 *
 * @param results - Every judged sample.
 * @returns The report lines, failures first.
 */
export function contrastReport(results: readonly ContrastResult[]): string[] {
  const lines: string[] = [];
  for (const result of results) {
    if (result.verdict !== "fail") continue;
    lines.push(
      `FAIL ${result.ratio.toFixed(2)}:1 (needs ${result.required}:1) ` +
        `${result.sample.label} — ${JSON.stringify(result.sample.text.slice(0, 40))}`,
    );
  }
  for (const result of results) {
    if (result.verdict !== "unresolved") continue;
    lines.push(
      `UNRESOLVED background ${result.sample.label} — ` +
        JSON.stringify(result.sample.text.slice(0, 40)),
    );
  }
  return lines;
}

/**
 * One colour pair that is known not to meet its threshold.
 *
 * Keyed on the **colours**, not on the element or its text. The first version
 * of this keyed on `label + text` and it was wrong twice over: the same
 * palette decision fails on dozens of unrelated elements, so the list said
 * "seventeen defects" where there were four colour pairs; and the text on
 * screen depends on which other tests ran first, so the same page produced 17
 * entries alone and 916 in a full suite. A baseline that changes with the
 * order of the suite is not a measurement.
 */
export interface ContrastBaselineEntry {
  /** Foreground as `r,g,b` after compositing. */
  foreground: string;
  /** Composited background as `r,g,b`. */
  background: string;
  ratio: number;
  required: number;
  /** Where it was first seen, for whoever goes looking. */
  seenAt: string;
}

/**
 * Return the key one colour pair is recorded under.
 *
 * Channels are rounded to whole numbers: a composite differing in the
 * thousandths is the same colour to every eye and every renderer.
 *
 * @param result - A judged sample with a resolved background.
 * @returns A stable key for the pair and its threshold.
 */
export function contrastKey(result: ContrastResult): string {
  const channels = (colour: Rgba): string =>
    `${Math.round(colour.r)},${Math.round(colour.g)},${Math.round(colour.b)}`;
  const background = result.sample.background;
  const foreground =
    background === null || result.sample.foreground.a === 1
      ? result.sample.foreground
      : composite(result.sample.foreground, background);
  return `${channels(foreground)} on ${background === null ? "?" : channels(background)} @${result.required}`;
}

/** How a run compares against the recorded failures. */
export interface ContrastComparison {
  /** Failures the baseline does not list: the palette got worse. */
  regressions: string[];
  /** Baseline entries that no longer fail: the list can shrink. */
  fixed: string[];
  /** Samples whose background could not be resolved, which are never scored. */
  unresolved: string[];
}

/**
 * Compare a run against the recorded failures.
 *
 * The list is exact in both directions. A failure it does not list is a
 * regression; an entry that no longer fails means the list is stale and can
 * come down. Allowing a stale entry to sit would let the baseline drift into a
 * blanket permission, which is the failure a ratchet exists to prevent.
 *
 * @param results - Every judged sample from one run.
 * @param baseline - The recorded failures.
 * @returns What is new, what is fixed, and what could not be measured.
 */
export function compareWithBaseline(
  results: readonly ContrastResult[],
  baseline: readonly ContrastBaselineEntry[],
): ContrastComparison {
  const recorded = new Set(
    baseline.map((entry) => `${entry.foreground} on ${entry.background} @${entry.required}`),
  );
  const failing = new Set<string>();
  // Keyed by the colour pair, so one palette decision is reported once however
  // many elements carry it; the message names the first element seen with it.
  const regressions = new Map<string, string>();
  const unresolved = new Set<string>();
  for (const result of results) {
    if (result.verdict === "unresolved") {
      unresolved.add(`${result.sample.label} — ${JSON.stringify(result.sample.text.slice(0, 40))}`);
      continue;
    }
    if (result.verdict !== "fail") continue;
    const id = contrastKey(result);
    failing.add(id);
    if (!recorded.has(id) && !regressions.has(id)) {
      regressions.set(
        id,
        `${result.ratio.toFixed(2)}:1 (needs ${result.required}:1) ${id} — ` +
          `${result.sample.label} ${JSON.stringify(result.sample.text.slice(0, 40))}`,
      );
    }
  }
  return {
    fixed: baseline
      .filter(
        (entry) =>
          !failing.has(`${entry.foreground} on ${entry.background} @${entry.required}`),
      )
      .map((entry) => `${entry.foreground} on ${entry.background} (${entry.seenAt})`),
    regressions: [...regressions.values()].sort(),
    unresolved: [...unresolved].sort(),
  };
}
