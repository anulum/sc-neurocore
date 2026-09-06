// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * A recording 2D context, for checking what a view drew.
 *
 * A canvas reports nothing about its contents, so the only way to assert on a
 * drawing is to record the calls that made it. Every text, rectangle and path
 * segment is kept in the order it was issued, along with the fill in force at
 * the time — which is what lets a case say a label was drawn in the axis
 * colour rather than merely that some text was drawn.
 */

import { vi } from "vitest";

/** One recorded piece of text and the fill it was drawn with. */
export interface RecordedText {
  /** The string that was drawn. */
  text: string;
  /** Where it was drawn, in CSS pixels. */
  x: number;
  /** Where it was drawn, in CSS pixels. */
  y: number;
  /** The `fillStyle` in force at the time. */
  fill: string;
}

/** One recorded rectangle and the fill it was drawn with. */
export interface RecordedRect {
  /** Left edge. */
  x: number;
  /** Top edge. */
  y: number;
  /** Width, which may be negative or zero. */
  width: number;
  /** Height. */
  height: number;
  /** The `fillStyle` in force at the time. */
  fill: string;
}

/** What a view drew, in the order it drew it. */
export interface PlotRecording {
  /** The context to pass to the view. */
  ctx: CanvasRenderingContext2D;
  /** Every `fillText` call. */
  texts: RecordedText[];
  /** Every `fillRect` call. */
  rects: RecordedRect[];
  /** Every path point, as `M x,y` or `L x,y`. */
  path: string[];
  /** Every `strokeStyle` that was set, in order. */
  strokes: string[];
}

/**
 * Build a recording context and the record it writes into.
 *
 * @returns The context and the arrays it records into, which fill as the view
 *   draws.
 */
export function mockPlotContext(): PlotRecording {
  const texts: RecordedText[] = [];
  const rects: RecordedRect[] = [];
  const path: string[] = [];
  const strokes: string[] = [];
  const state = { fillStyle: "", strokeStyle: "" };
  const ctx = {
    get fillStyle() {
      return state.fillStyle;
    },
    set fillStyle(value: string) {
      state.fillStyle = value;
    },
    get strokeStyle() {
      return state.strokeStyle;
    },
    set strokeStyle(value: string) {
      state.strokeStyle = value;
      strokes.push(value);
    },
    lineWidth: 0,
    font: "",
    textAlign: "start",
    fillRect: vi.fn((x: number, y: number, width: number, height: number) => {
      rects.push({ fill: state.fillStyle, height, width, x, y });
    }),
    strokeRect: vi.fn(),
    beginPath: vi.fn(),
    closePath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    moveTo: vi.fn((x: number, y: number) => {
      path.push(`M${String(x)},${String(y)}`);
    }),
    lineTo: vi.fn((x: number, y: number) => {
      path.push(`L${String(x)},${String(y)}`);
    }),
    stroke: vi.fn(),
    setLineDash: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn(),
    rotate: vi.fn(),
    scale: vi.fn(),
    fillText: vi.fn((text: string, x: number, y: number) => {
      texts.push({ fill: state.fillStyle, text, x, y });
    }),
  } as unknown as CanvasRenderingContext2D;
  return { ctx, path, rects, strokes, texts };
}

/**
 * Whether any recorded number is not finite.
 *
 * A `NaN` coordinate draws nothing and reports no error, so a view that
 * computes one fails silently. Every case here checks for it, because it is
 * the failure this kind of code actually has.
 *
 * @param recording - The record to inspect.
 * @returns Whether anything drawn carried a non-finite coordinate.
 */
export function drewNonFinite(recording: PlotRecording): boolean {
  const numbers = [
    ...recording.rects.flatMap((r) => [r.x, r.y, r.width, r.height]),
    ...recording.texts.flatMap((t) => [t.x, t.y]),
    ...recording.path.map((point) => Number(point.slice(1).split(",")[0])),
    ...recording.path.map((point) => Number(point.slice(1).split(",")[1])),
  ];
  return numbers.some((value) => !Number.isFinite(value));
}
