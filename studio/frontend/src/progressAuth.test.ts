// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio WebSocket authentication tests

import { describe, expect, it } from "vitest";

import { progressWebSocketProtocols, setStudioAuthToken } from "./api/client";

describe("progress WebSocket authentication", () => {
  it("omits WebSocket protocols when no bearer token is active", () => {
    setStudioAuthToken(null);

    expect(progressWebSocketProtocols()).toBeUndefined();
  });

  it("encodes the active bearer token as a browser WebSocket subprotocol", () => {
    setStudioAuthToken("session-token");

    expect(progressWebSocketProtocols()).toEqual([
      "studio-auth",
      "studio-bearer.session-token",
    ]);
  });

  it("accepts an explicit token without mutating global auth state", () => {
    setStudioAuthToken(null);

    expect(progressWebSocketProtocols("temporary-token")).toEqual([
      "studio-auth",
      "studio-bearer.temporary-token",
    ]);
    expect(progressWebSocketProtocols()).toBeUndefined();
  });
});
