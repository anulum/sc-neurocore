// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio share URL browser runtime tests

import { describe, expect, it } from "vitest";

import {
  STUDIO_SHARE_STATUS_CLEAR_DELAY_MS,
  copyStudioShareUrlInRuntime,
  scheduleStudioShareStatusClear,
  type StudioShareStatusClearScheduler,
  type StudioShareStatusClearTimer,
  studioShareStatusClearedState,
  studioShareStatusState,
  type StudioShareRuntime,
} from "./studioShareRuntime";
import type { StudioShareUrlInput } from "./studioUrlState";

const input: StudioShareUrlInput = {
  current: 10,
  dt: 0.1,
  duration: 100,
  equations: ["dv/dt = -v / tau"],
  modelParams: { tau: 10 },
  odeInit: { v: 0 },
  odeParams: { tau: 20 },
  protocol: "constant",
  reset: "v = 0",
  selectedModelName: "lif",
  sourceMode: "model",
  threshold: "v > 1",
};

class RecordingClearScheduler implements StudioShareStatusClearScheduler {
  readonly delays: number[] = [];
  private callbacks: Array<() => void> = [];

  setTimeout(callback: () => void, delayMs: number): StudioShareStatusClearTimer {
    this.callbacks.push(callback);
    this.delays.push(delayMs);
    return this.callbacks.length as unknown as StudioShareStatusClearTimer;
  }

  runLatest(): void {
    this.callbacks[this.callbacks.length - 1]?.();
  }
}

describe("Studio share URL browser runtime", () => {
  it("copies the generated URL through the runtime clipboard", async () => {
    const writes: string[] = [];
    const runtime: StudioShareRuntime = {
      clipboard: {
        writeText: (text) => {
          writes.push(text);
          return Promise.resolve();
        },
      },
      location: { origin: "https://studio.example", pathname: "/workbench" },
    };

    const result = await copyStudioShareUrlInRuntime(input, runtime, (payload) => `encoded:${payload}`);

    expect(result).toEqual({
      ok: true,
      url: writes[0],
    });
    expect(writes).toHaveLength(1);
    expect(writes[0]).toContain("https://studio.example/workbench#encoded:");
    expect(writes[0]).toContain("\"mn\":\"lif\"");
  });

  it("reports a non-browser runtime instead of touching browser globals", async () => {
    await expect(copyStudioShareUrlInRuntime(input, null)).resolves.toEqual({
      ok: false,
      message: "Share URL is available only in a browser session.",
    });
  });

  it("reports missing clipboard access", async () => {
    await expect(copyStudioShareUrlInRuntime(input, {
      clipboard: null,
      location: { origin: "https://studio.example", pathname: "/workbench" },
    })).resolves.toEqual({
      ok: false,
      message: "Clipboard access is unavailable in this browser session.",
    });
  });

  it("reports clipboard write failures", async () => {
    await expect(copyStudioShareUrlInRuntime(input, {
      clipboard: {
        writeText: () => Promise.reject(new Error("clipboard denied")),
      },
      location: { origin: "https://studio.example", pathname: "/workbench" },
    })).resolves.toEqual({
      ok: false,
      message: "clipboard denied",
    });
  });

  it("builds the success status patch for store consumers", () => {
    expect(studioShareStatusState({ ok: true, url: "https://studio.example/workbench#state" }))
      .toEqual({ error: "URL copied to clipboard" });
  });

  it("builds the failure status patch for store consumers", () => {
    expect(studioShareStatusState({ ok: false, message: "Clipboard access is unavailable." }))
      .toEqual({ error: "Clipboard access is unavailable." });
  });

  it("builds the cleared status patch for store consumers", () => {
    expect(studioShareStatusClearedState()).toEqual({ error: null });
  });

  it("schedules share status clearing with the canonical delay", () => {
    const scheduler = new RecordingClearScheduler();
    let clearCount = 0;

    scheduleStudioShareStatusClear(() => {
      clearCount += 1;
    }, scheduler);
    scheduler.runLatest();

    expect(scheduler.delays).toEqual([STUDIO_SHARE_STATUS_CLEAR_DELAY_MS]);
    expect(clearCount).toBe(1);
  });
});
