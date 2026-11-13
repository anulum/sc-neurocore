// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tests for the render timing marks

import { afterEach, describe, expect, expectTypeOf, it } from "vitest";

import { MAX_MARKS_PER_NAME, RENDER_MARK_PREFIX, RENDER_MARKS, markRender, renderMarkName } from "./renderTiming";
import type { RenderMark } from "./renderTiming";

afterEach(() => {
  performance.clearMarks();
});

/** The marks of one name the page has recorded so far. */
function recorded(name: string): PerformanceEntry[] {
  return performance.getEntriesByName(name, "mark");
}

describe("render timing marks", () => {
  it("names six fixed marks under one prefix", () => {
    expect(RENDER_MARK_PREFIX).toBe("studio:");
    expect(RENDER_MARKS).toEqual([
      "catalogue-drawn",
      "trace-drawn",
      "view-drawn",
      "network-drawn",
      "training-event-received",
      "training-chart-drawn",
    ]);
    expect(RENDER_MARKS.map(renderMarkName)).toEqual(RENDER_MARKS.map((name) => `studio:${name}`));
  });

  it("emits the named mark each time it is called", () => {
    for (const name of RENDER_MARKS) {
      markRender(name);
      markRender(name);
      expect(recorded(`studio:${name}`)).toHaveLength(2);
    }
  });

  it("takes only a listed name, as a type and at run time", () => {
    expectTypeOf(markRender).parameter(0).toEqualTypeOf<RenderMark>();
    expect(() => {
      markRender("trace-painted" as RenderMark);
    }).toThrow("trace-painted is not a render timing mark; the marks are catalogue-drawn, trace-drawn, view-drawn, network-drawn, training-event-received, training-chart-drawn");
    expect(recorded("studio:trace-painted")).toHaveLength(0);
  });

  it("keeps at most a bounded number of marks of one name, so a long session does not grow the timeline", () => {
    expect(MAX_MARKS_PER_NAME).toBe(1000);
    for (let index = 0; index < MAX_MARKS_PER_NAME; index += 1) markRender("network-drawn");
    expect(recorded("studio:network-drawn")).toHaveLength(MAX_MARKS_PER_NAME);
    markRender("trace-drawn");
    markRender("network-drawn");
    expect(recorded("studio:network-drawn")).toHaveLength(1);
    expect(recorded("studio:trace-drawn")).toHaveLength(1);
  });

  it("does nothing where the page offers no performance timeline", () => {
    const original = performance.mark.bind(performance);
    let calls = 0;
    performance.mark = () => {
      calls += 1;
      throw new Error("no timeline");
    };
    try {
      expect(() => {
        markRender("view-drawn");
      }).not.toThrow();
      expect(calls).toBe(1);
    } finally {
      performance.mark = original;
    }
  });
});
