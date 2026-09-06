// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The contrast measurement, against the specification's own numbers

/**
 * A contrast check that looks at the wrong pixels passes for the wrong reason.
 *
 * The two ways to get this wrong are both silent. Reading a background from
 * the element itself finds `rgba(0,0,0,0)` almost everywhere and scores text
 * against transparency; assuming white when the stack cannot be resolved
 * manufactures a pass for dark text on an unknown ground. These cases pin the
 * compositing, and pin that an unresolved background is reported rather than
 * scored.
 *
 * The arithmetic is checked against values the WCAG specification states
 * itself — black on white is 21:1, a colour against itself is 1:1 — rather
 * than against this implementation's own output.
 */

import { describe, expect, it } from "vitest";

import {
  CONTRAST_AA_LARGE,
  CONTRAST_AA_NORMAL,
  compareWithBaseline,
  composite,
  contrastKey,
  contrastRatio,
  contrastReport,
  judge,
  parseRgba,
  relativeLuminance,
  requiredRatio,
  resolveBackground,
  type ContrastSample,
} from "./contrastAudit";

const BLACK = { a: 1, b: 0, g: 0, r: 0 };
const WHITE = { a: 1, b: 255, g: 255, r: 255 };

/** One sample, overridden where a case needs a particular value. */
function sample(overrides: Partial<ContrastSample> = {}): ContrastSample {
  return {
    background: WHITE,
    bold: false,
    fontSize: 12,
    foreground: BLACK,
    label: "button",
    text: "Simulate",
    ...overrides,
  };
}

describe("reading a computed colour", () => {
  it("reads the two forms a browser reports", () => {
    expect(parseRgba("rgb(18, 52, 86)")).toEqual({ a: 1, b: 86, g: 52, r: 18 });
    expect(parseRgba("rgba(18, 52, 86, 0.5)")).toEqual({ a: 0.5, b: 86, g: 52, r: 18 });
  });

  it("reads the space-separated form as well", () => {
    expect(parseRgba("rgb(18 52 86 / 0.25)")).toEqual({ a: 0.25, b: 86, g: 52, r: 18 });
  });

  it("refuses anything it cannot read rather than guessing", () => {
    // A gradient hides what is behind it and has no single colour.
    expect(parseRgba("linear-gradient(red, blue)")).toBeNull();
    expect(parseRgba("transparent")).toBeNull();
    expect(parseRgba("")).toBeNull();
  });
});

describe("resolving the background a reader sees", () => {
  it("takes the first opaque layer when nothing above it paints", () => {
    expect(resolveBackground([{ a: 0, b: 0, g: 0, r: 0 }, WHITE])).toEqual(WHITE);
  });

  it("paints translucent layers over the opaque one, in order", () => {
    // 50% black over white is mid grey.
    const resolved = resolveBackground([{ a: 0.5, b: 0, g: 0, r: 0 }, WHITE]);

    expect(resolved?.r).toBeCloseTo(127.5, 5);
    expect(resolved?.a).toBe(1);
  });

  it("composites several translucent layers in paint order", () => {
    const resolved = resolveBackground([
      { a: 0.5, b: 0, g: 0, r: 0 },
      { a: 0.5, b: 255, g: 255, r: 255 },
      BLACK,
    ]);

    // Inner 50% white over black is 127.5; 50% black over that is 63.75.
    expect(resolved?.r).toBeCloseTo(63.75, 5);
  });

  it("reports nothing when the stack never reaches an opaque colour", () => {
    // Assuming white here would manufacture a pass for dark text.
    expect(resolveBackground([{ a: 0.5, b: 0, g: 0, r: 0 }])).toBeNull();
    expect(resolveBackground([])).toBeNull();
  });

  it("reports nothing when a layer cannot be read at all", () => {
    expect(resolveBackground([null, WHITE])).toBeNull();
  });

  it("paints one colour over another as a browser does", () => {
    const painted = composite({ a: 0.25, b: 0, g: 0, r: 0 }, WHITE);

    expect(painted).toEqual({ a: 1, b: 191.25, g: 191.25, r: 191.25 });
  });
});

describe("the specification's own arithmetic", () => {
  it("gives black on white as 21 to 1", () => {
    expect(contrastRatio(BLACK, WHITE)).toBeCloseTo(21, 5);
  });

  it("gives a colour against itself as 1 to 1", () => {
    expect(contrastRatio(WHITE, WHITE)).toBeCloseTo(1, 10);
  });

  it("is symmetric in its arguments", () => {
    const grey = { a: 1, b: 119, g: 119, r: 119 };

    expect(contrastRatio(grey, WHITE)).toBeCloseTo(contrastRatio(WHITE, grey), 10);
  });

  it("linearises the sRGB channels rather than using them raw", () => {
    // Mid grey has a luminance well below 0.5 once linearised; a raw average
    // would put it at about 0.5 and pass contrasts that fail.
    expect(relativeLuminance({ a: 1, b: 128, g: 128, r: 128 })).toBeCloseTo(0.2158, 3);
  });
});

describe("which threshold applies", () => {
  it("asks 4.5 to 1 of body text", () => {
    expect(requiredRatio({ bold: false, fontSize: 12 })).toBe(CONTRAST_AA_NORMAL);
  });

  it("asks 3 to 1 of large text", () => {
    expect(requiredRatio({ bold: false, fontSize: 24 })).toBe(CONTRAST_AA_LARGE);
  });

  it("counts bold text as large from 18.66px", () => {
    expect(requiredRatio({ bold: true, fontSize: 18.66 })).toBe(CONTRAST_AA_LARGE);
    expect(requiredRatio({ bold: true, fontSize: 18 })).toBe(CONTRAST_AA_NORMAL);
  });

  it("does not relax the threshold for non-bold text below 24px", () => {
    expect(requiredRatio({ bold: false, fontSize: 23.9 })).toBe(CONTRAST_AA_NORMAL);
  });
});

describe("judging one sample", () => {
  it("passes black on white", () => {
    const result = judge(sample());

    expect(result.verdict).toBe("pass");
    expect(result.ratio).toBe(21);
  });

  it("fails a contrast the palette does not meet", () => {
    const result = judge(sample({ foreground: { a: 1, b: 153, g: 153, r: 153 } }));

    expect(result.verdict).toBe("fail");
    if (result.verdict === "unresolved") throw new Error("expected a measured verdict");
    expect(result.ratio).toBeLessThan(CONTRAST_AA_NORMAL);
  });

  it("composites a translucent foreground over its own background", () => {
    // Text at 50% opacity is not the colour it declares.
    const opaque = judge(sample({ foreground: BLACK }));
    const faded = judge(sample({ foreground: { a: 0.5, b: 0, g: 0, r: 0 } }));
    if (opaque.ratio === null || faded.ratio === null) {
      throw new Error("expected both to be measured");
    }

    expect(faded.ratio).toBeLessThan(opaque.ratio);
  });

  it("reports an unresolved background rather than scoring it", () => {
    const result = judge(sample({ background: null }));

    expect(result.verdict).toBe("unresolved");
    expect(result.ratio).toBeNull();
  });

  it("uses the large-text threshold when the sample is large", () => {
    const grey = { a: 1, b: 128, g: 128, r: 128 };

    expect(judge(sample({ foreground: grey })).verdict).toBe("fail");
    expect(judge(sample({ fontSize: 24, foreground: grey })).verdict).toBe("pass");
  });

  it("does not fail a sample by a difference its own report cannot show", () => {
    // The report prints two places; a value that prints as the threshold is
    // not failed by digits beyond it.
    const result = judge(sample({ foreground: { a: 1, b: 117, g: 117, r: 117 } }));
    if (result.ratio === null) throw new Error("expected a measured verdict");

    expect(result.ratio).toBe(Math.round(result.ratio * 100) / 100);
  });
});

describe("the report", () => {
  it("names the ratio, the threshold and the element that failed", () => {
    const lines = contrastReport([judge(sample({ foreground: { a: 1, b: 153, g: 153, r: 153 } }))]);

    expect(lines).toHaveLength(1);
    expect(lines[0]).toContain("FAIL");
    expect(lines[0]).toContain("needs 4.5:1");
    expect(lines[0]).toContain("button");
    expect(lines[0]).toContain("Simulate");
  });

  it("reports an unresolved background separately from a failure", () => {
    const lines = contrastReport([judge(sample({ background: null }))]);

    expect(lines[0]).toContain("UNRESOLVED");
  });

  it("says nothing about the samples that passed", () => {
    expect(contrastReport([judge(sample())])).toEqual([]);
  });

  it("puts failures before unresolved backgrounds", () => {
    const lines = contrastReport([
      judge(sample({ background: null })),
      judge(sample({ foreground: { a: 1, b: 153, g: 153, r: 153 } })),
    ]);

    expect(lines[0]).toContain("FAIL");
    expect(lines[1]).toContain("UNRESOLVED");
  });
});

describe("comparing a run against the recorded failures", () => {
  const GREY = { a: 1, b: 153, g: 153, r: 153 };
  const failing = sample({ foreground: GREY, label: "button", text: "Undo" });
  const baseline = [
    {
      background: "255,255,255",
      foreground: "153,153,153",
      ratio: 2.32,
      required: 4.5,
      seenAt: "toolbar",
    },
  ];

  it("keys on the colour pair, not on the element or its text", () => {
    // One palette decision fails on many unrelated elements; keying on text
    // would call that many defects, and would change with whatever else the
    // suite left on screen.
    const elsewhere = sample({ foreground: GREY, label: "span", text: "Something else" });

    const comparison = compareWithBaseline([judge(failing), judge(elsewhere)], baseline);

    expect(comparison.regressions).toEqual([]);
    expect(comparison.fixed).toEqual([]);
  });

  it("reports a colour pair the baseline does not record", () => {
    const worse = sample({ foreground: { a: 1, b: 200, g: 200, r: 200 }, label: "span", text: "New" });

    const comparison = compareWithBaseline([judge(failing), judge(worse)], baseline);

    expect(comparison.regressions).toHaveLength(1);
    expect(comparison.regressions[0]).toContain("200,200,200 on 255,255,255");
  });

  it("reports the same new pair once however many elements carry it", () => {
    const worse = { a: 1, b: 200, g: 200, r: 200 };

    const comparison = compareWithBaseline(
      [
        judge(sample({ foreground: worse, label: "span", text: "One" })),
        judge(sample({ foreground: worse, label: "div", text: "Two" })),
      ],
      baseline,
    );

    expect(comparison.regressions).toHaveLength(1);
  });

  it("separates the same colours at different thresholds", () => {
    // 120-grey on white is 4.42:1 — under 4.5 for body text, over 3 for large.
    const colour = { a: 1, b: 120, g: 120, r: 120 };
    const body = sample({ foreground: colour, label: "span", text: "Body" });
    const large = sample({ fontSize: 24, foreground: colour, label: "h1", text: "Title" });

    expect(judge(body).verdict).toBe("fail");
    expect(judge(large).verdict).toBe("pass");
    // Only the body-text failure is reported; the large one is not a defect.
    expect(compareWithBaseline([judge(body), judge(large)], baseline).regressions).toHaveLength(1);
  });

  it("reports a baseline entry that no longer fails, so the list can come down", () => {
    const comparison = compareWithBaseline([judge(sample({ label: "button", text: "Undo" }))], baseline);

    expect(comparison.fixed).toHaveLength(1);
    expect(comparison.fixed[0]).toContain("toolbar");
  });

  it("reports an unresolved background rather than counting it either way", () => {
    const comparison = compareWithBaseline(
      [judge(sample({ background: null, label: "div", text: "Unknown" }))],
      [],
    );

    expect(comparison.unresolved).toHaveLength(1);
    expect(comparison.regressions).toEqual([]);
  });

  it("says nothing at all when a run passes and the baseline is empty", () => {
    expect(compareWithBaseline([judge(sample())], [])).toEqual({
      fixed: [],
      regressions: [],
      unresolved: [],
    });
  });
});

describe("the key a colour pair is recorded under", () => {
  it("rounds the channels, because a thousandth is the same colour", () => {
    const key = contrastKey(
      judge(sample({ background: { a: 1, b: 254.6, g: 254.6, r: 254.6 } })),
    );

    expect(key).toContain("on 255,255,255");
  });

  it("composites a translucent foreground before keying it", () => {
    const key = contrastKey(judge(sample({ foreground: { a: 0.5, b: 0, g: 0, r: 0 } })));

    expect(key.startsWith("128,128,128")).toBe(true);
  });

  it("carries the threshold, so one pair is two entries at two sizes", () => {
    const small = contrastKey(judge(sample({ foreground: BLACK })));
    const large = contrastKey(judge(sample({ fontSize: 24, foreground: BLACK })));

    expect(small).not.toBe(large);
  });
});
