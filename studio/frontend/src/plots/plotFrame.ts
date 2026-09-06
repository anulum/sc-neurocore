// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The geometry every Studio plot view draws into.
 *
 * The views were branches of one 836-line component, each reading the same
 * half-dozen local numbers off its enclosing scope. Passing them as one value
 * is what let the views become functions that can be called — and tested —
 * without mounting anything.
 */

import { PLOT_BG } from "../simulationPlotCanvas";

/** Inset from the left, leaving room for the y-axis labels. */
export const PLOT_LEFT = 52;

/** Inset from the right. */
export const PLOT_RIGHT = 12;

/** Inset from the top. */
export const PLOT_TOP = 8;

/** Inset from the bottom, leaving room for the x-axis labels. */
export const PLOT_BOTTOM = 18;

/** Below this, in CSS pixels, there is not enough room to draw anything. */
export const PLOT_MIN_SIDE = 100;

/** Where one plot view may draw, in CSS pixels. */
export interface PlotFrame {
  /** Left edge of the plot area. */
  left: number;
  /** Top edge of the plot area. */
  top: number;
  /** Height reserved below the plot area for x-axis labels. */
  bottom: number;
  /** Width of the plot area, between the left and right insets. */
  plotWidth: number;
  /** Whole canvas width. */
  width: number;
  /** Whole canvas height. */
  height: number;
}

/**
 * Build a frame for a canvas of this size.
 *
 * @param width - Canvas width in CSS pixels.
 * @param height - Canvas height in CSS pixels.
 * @returns The frame the views draw into.
 */
export function plotFrame(width: number, height: number): PlotFrame {
  return {
    bottom: PLOT_BOTTOM,
    height,
    left: PLOT_LEFT,
    plotWidth: width - PLOT_LEFT - PLOT_RIGHT,
    top: PLOT_TOP,
    width,
  };
}

/**
 * Size a canvas to its container, scale it for the display, and clear it.
 *
 * The canvas is sized in device pixels and scaled back by the device pixel
 * ratio, so a line asked for at one pixel is one CSS pixel wide on any display
 * rather than a blurred one and a half.
 *
 * @param canvas - The canvas to size.
 * @param width - Container width in CSS pixels.
 * @param height - Container height in CSS pixels.
 * @param devicePixelRatio - The display's ratio, at least 1.
 * @returns The prepared context and frame, or `null` when there is no room to
 *   draw or the browser refuses a 2D context.
 */
export function preparePlotCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  devicePixelRatio: number,
): { ctx: CanvasRenderingContext2D; frame: PlotFrame } | null {
  if (width < PLOT_MIN_SIDE || height < PLOT_MIN_SIDE) return null;
  const ratio = devicePixelRatio > 0 ? devicePixelRatio : 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.width = `${String(width)}px`;
  canvas.style.height = `${String(height)}px`;
  const ctx = canvas.getContext("2d");
  if (ctx === null) return null;
  ctx.scale(ratio, ratio);
  ctx.fillStyle = PLOT_BG;
  ctx.fillRect(0, 0, width, height);
  return { ctx, frame: plotFrame(width, height) };
}
