// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — render timing marks: fixed names on the browser's performance timeline

/**
 * Put a fixed, named mark on the performance timeline when a view has drawn.
 *
 * The Studio measures how long it takes to show a result: the live timing
 * spec (`e2e/render-timing-live.spec.ts`) puts a start mark just before a user
 * action and reads the first of these marks after it. The marks change
 * nothing on screen and carry no data, only their name and the time; they are
 * always on and read only by tests.
 */

/** The prefix every render timing mark carries on the timeline. */
export const RENDER_MARK_PREFIX = "studio:";

/** The marks, one per place that draws a measured result. */
export const RENDER_MARKS = [
  "catalogue-drawn",
  "trace-drawn",
  "view-drawn",
  "network-drawn",
  "training-event-received",
  "training-chart-drawn",
] as const;

/** One render timing mark. */
export type RenderMark = (typeof RENDER_MARKS)[number];

/**
 * The most marks of one name the timeline keeps. A dragged network node marks every
 * frame; past this many the older marks of that name are cleared, so a long
 * session does not grow the timeline without bound.
 */
export const MAX_MARKS_PER_NAME = 1000;

/**
 * Return the name a mark carries on the timeline.
 *
 * @param mark - The mark.
 * @returns The prefixed name.
 */
export function renderMarkName(mark: RenderMark): string {
  return `${RENDER_MARK_PREFIX}${mark}`;
}

/**
 * Mark that a measured result was drawn, or a measured event arrived.
 *
 * @param mark - One of {@link RENDER_MARKS}.
 * @throws {Error} When the name is not a render timing mark: a misspelt name
 *   would leave its measure without an end, so it is refused rather than
 *   recorded under another name.
 */
export function markRender(mark: RenderMark): void {
  if (!(RENDER_MARKS as readonly string[]).includes(mark)) {
    throw new Error(`${mark} is not a render timing mark; the marks are ${RENDER_MARKS.join(", ")}`);
  }
  const name = renderMarkName(mark);
  try {
    if (performance.getEntriesByName(name, "mark").length >= MAX_MARKS_PER_NAME) {
      performance.clearMarks(name);
    }
    performance.mark(name);
  } catch {
    // Instrumentation never stops a view from drawing: a page without a
    // usable timeline simply has no measure.
  }
}
