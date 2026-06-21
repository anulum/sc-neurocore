// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio auto-simulation debounce scheduler

export const STUDIO_AUTO_SIMULATE_DELAY_MS = 250;

export type StudioAutoSimulationTimer = ReturnType<typeof setTimeout>;

export interface StudioAutoSimulationScheduler {
  clearTimeout(timer: StudioAutoSimulationTimer): void;
  setTimeout(callback: () => void, delayMs: number): StudioAutoSimulationTimer;
}

export function browserAutoSimulationScheduler(): StudioAutoSimulationScheduler {
  return {
    clearTimeout: (timer) => clearTimeout(timer),
    setTimeout: (callback, delayMs) => setTimeout(callback, delayMs),
  };
}

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
