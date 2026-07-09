// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AuthControlView } from "./AuthControl";

describe("AuthControl", () => {
  it("renders the browser login form when no session is authenticated", () => {
    const html = renderToStaticMarkup(
      <AuthControlView
        authError={null}
        authLoading={false}
        authSession={{ authenticated: false, principal_id: null, roles: [] }}
        onLogin={async () => undefined}
        onLogout={async () => undefined}
      />,
    );

    expect(html).toContain("Studio username");
    expect(html).toContain("Studio password");
    expect(html).toContain("Login");
  });

  it("renders the current principal and logout action for authenticated sessions", () => {
    const html = renderToStaticMarkup(
      <AuthControlView
        authError={null}
        authLoading={false}
        authSession={{
          authenticated: true,
          principal_id: "user-operator",
          roles: ["studio.admin"],
        }}
        onLogin={async () => undefined}
        onLogout={async () => undefined}
      />,
    );

    expect(html).toContain("user-operator");
    expect(html).toContain("Logout");
    expect(html).not.toContain("password");
  });
});
