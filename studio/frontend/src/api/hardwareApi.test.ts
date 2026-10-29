// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hardware discovery shared HTTP transport tests

import { afterEach, expect, it, vi } from "vitest";
import { fetchHardwareCatalogue } from "./client";
import { setStudioAuthToken } from "./http";

afterEach(() => {
  setStudioAuthToken(null);
  vi.unstubAllGlobals();
});

it("uses the same authenticated JSON transport as the workbench", async () => {
  const payload = { schema_version: "studio.hardware-catalogue.v1", entries: [], evidence_scope: "Registration only" };
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), { status: 200 }));
  vi.stubGlobal("fetch", fetcher);
  setStudioAuthToken("catalogue-fixture");
  await expect(fetchHardwareCatalogue()).resolves.toEqual(payload);
  expect(fetcher).toHaveBeenCalledWith("/api/hardware/catalogue", { headers: { Authorization: "Bearer catalogue-fixture" } });
});

it("preserves a real HTTP refusal instead of inventing empty registrations", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Discovery refused" }), { status: 503 })));
  await expect(fetchHardwareCatalogue()).rejects.toThrow("Discovery refused");
});
