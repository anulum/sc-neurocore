// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio auto-simulation debounce scheduler tests

import { describe, expect, it } from "vitest";

import {
  STUDIO_AUTO_SIMULATE_DELAY_MS,
  scheduleStudioAutoSimulation,
  type StudioAutoSimulationScheduler,
  type StudioAutoSimulationTimer,
} from "./studioAutoSimulation";

class RecordingScheduler implements StudioAutoSimulationScheduler {
  readonly clearedTimers: StudioAutoSimulationTimer[] = [];
  readonly delays: number[] = [];
  private callbacks: Array<() => void> = [];

  clearTimeout(timer: StudioAutoSimulationTimer): void {
    this.clearedTimers.push(timer);
  }

  setTimeout(callback: () => void, delayMs: number): StudioAutoSimulationTimer {
    this.callbacks.push(callback);
    this.delays.push(delayMs);
    return this.callbacks.length as unknown as StudioAutoSimulationTimer;
  }

  runLatest(): void {
    this.callbacks[this.callbacks.length - 1]?.();
  }
}

describe("Studio auto-simulation debounce scheduler", () => {
  it("schedules auto-simulation with the canonical debounce delay", () => {
    const scheduler = new RecordingScheduler();
    let runs = 0;

    scheduleStudioAutoSimulation(null, () => {
      runs += 1;
    }, scheduler);
    scheduler.runLatest();

    expect(scheduler.clearedTimers).toEqual([]);
    expect(scheduler.delays).toEqual([STUDIO_AUTO_SIMULATE_DELAY_MS]);
    expect(runs).toBe(1);
  });

  it("clears a pending auto-simulation timer before scheduling the next one", () => {
    const scheduler = new RecordingScheduler();
    const firstTimer = scheduleStudioAutoSimulation(null, () => undefined, scheduler);

    const secondTimer = scheduleStudioAutoSimulation(
      firstTimer,
      () => undefined,
      scheduler,
      50,
    );

    expect(secondTimer).not.toBe(firstTimer);
    expect(scheduler.clearedTimers).toEqual([firstTimer]);
    expect(scheduler.delays).toEqual([STUDIO_AUTO_SIMULATE_DELAY_MS, 50]);
  });
});
