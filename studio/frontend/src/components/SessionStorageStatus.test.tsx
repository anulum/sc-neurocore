// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Session persistence status rendering
// @vitest-environment happy-dom

import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import SessionStorageStatus from "./SessionStorageStatus";
import type { StudioSessionWriteStatus } from "../studioSavedSessions";

it("renders no durability claim before a session write", () => {
  expect(renderToStaticMarkup(<SessionStorageStatus result={null} />)).toBe("");
});

it("announces a successful browser write without a failure warning", () => {
  const html = renderToStaticMarkup(<SessionStorageStatus result={{ status: "ok", message: "" }} />);
  const container = document.createElement("div");
  container.innerHTML = html;
  expect(container.querySelector('[role="status"]')?.textContent).toContain("saved in this browser");
  expect(container.querySelector('[role="alert"]')).toBeNull();
  expect(container.textContent).not.toContain("this page only");
});

it.each<StudioSessionWriteStatus>(["unavailable", "quota-exceeded", "failed"])(
  "explains page-only changes and reload recovery for %s",
  (status) => {
    const container = document.createElement("div");
    container.innerHTML = renderToStaticMarkup(
      <SessionStorageStatus result={{ status, message: "Storage refused <this> write." }} />,
    );
    const notice = container.querySelector('[role="alert"]');
    expect(notice?.textContent).toContain("Storage refused <this> write.");
    expect(notice?.textContent).toContain("this page only");
    expect(notice?.textContent).toContain("Reloading may restore the last saved list");
    expect(notice?.textContent).toContain("Save a server project");
    expect(notice?.querySelector("this")).toBeNull();
    expect(container.querySelector('[role="status"]')).toBeNull();
  },
);
