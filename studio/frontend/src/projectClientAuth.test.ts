// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio project client authentication tests

import { afterEach, describe, expect, it, vi } from "vitest";

import { deleteProject, setStudioAuthToken } from "./api/client";

describe("project client authentication", () => {
  afterEach(() => {
    setStudioAuthToken(null);
    vi.restoreAllMocks();
  });

  it("sends the active bearer token when deleting a saved project", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ deleted: "demo" }), { status: 200 }));
    setStudioAuthToken("session-token");

    await expect(deleteProject("demo")).resolves.toEqual({ deleted: "demo" });

    expect(fetchMock).toHaveBeenCalledWith("/api/project/demo", {
      headers: { Authorization: "Bearer session-token" },
      method: "DELETE",
    });
  });

  it("omits bearer headers when no Studio token is active", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ deleted: "demo" }), { status: 200 }));

    await expect(deleteProject("demo")).resolves.toEqual({ deleted: "demo" });

    expect(fetchMock).toHaveBeenCalledWith("/api/project/demo", {
      headers: {},
      method: "DELETE",
    });
  });
});
