// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio auto-simulation debounce scheduler

/**
 * Re-running the simulation shortly after the reader stops changing things.
 *
 * A quarter of a second: long enough that dragging a slider does not submit a
 * run per pixel, short enough that a deliberate edit feels answered. The timer
 * is injectable so a test can advance it rather than wait for it.
 */

/** How long to wait after the last change before re-running. */
export const STUDIO_AUTO_SIMULATE_DELAY_MS = 250;

/** The handle a scheduled run returns, so the next change can cancel it. */
export type StudioAutoSimulationTimer = ReturnType<typeof setTimeout>;

/**
 * The two timer calls this module makes. Declared rather than calling the
 * globals, so a test can run the delay without waiting for it.
 */
export interface StudioAutoSimulationScheduler {
  clearTimeout(timer: StudioAutoSimulationTimer): void;
  setTimeout(callback: () => void, delayMs: number): StudioAutoSimulationTimer;
}

/**
 * The browser's own timer.
 *
 * @returns The scheduler.
 */
export function browserAutoSimulationScheduler(): StudioAutoSimulationScheduler {
  return {
    clearTimeout: (timer) => {
      clearTimeout(timer);
    },
    setTimeout: (callback, delayMs) => setTimeout(callback, delayMs),
  };
}

/**
 * Schedule a run, cancelling any run already waiting.
 *
 * Cancelling first is what makes this a debounce rather than a queue: the
 * reader wants one run of what they finally set, not one per keystroke.
 *
 * @param currentTimer - The run already waiting, if any.
 * @param runSimulation - What to run when the delay elapses.
 * @param scheduler - The scheduler; the browser's own by default.
 * @param delayMs - How long to wait.
 * @returns The new handle, for the next change to cancel.
 */
export function scheduleStudioAutoSimulation(
  currentTimer: StudioAutoSimulationTimer | null,
  runSimulation: () => void,
  scheduler: StudioAutoSimulationScheduler = browserAutoSimulationScheduler(),
  delayMs: number = STUDIO_AUTO_SIMULATE_DELAY_MS,
): StudioAutoSimulationTimer {
  if (currentTimer !== null) {
    scheduler.clearTimeout(currentTimer);
  }
  return scheduler.setTimeout(runSimulation, delayMs);
}
