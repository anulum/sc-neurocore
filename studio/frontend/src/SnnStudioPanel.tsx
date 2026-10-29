// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — federated Studio panel entry

import { lazy, Suspense } from "react";
import HostedStudioPanel from "./HostedStudioPanel";
import "./index.css";

const StandaloneStudioPanel = lazy(() => import("./StandaloneStudioPanel"));

/**
 * Mount the hosted catalogue and optional local runtime in an iframe, or standalone Studio.
 *
 * Embedded panels keep the standalone workbench, editor and store unmounted.
 * Metadata loads first; explicit local execution uses the separately declared
 * closed host capability and its operator-configured backend.
 *
 * @returns The hosted catalogue/runtime view or the standalone Studio workspace.
 */
export default function SnnStudioPanel() {
  if (window.parent !== window) return <HostedStudioPanel />;
  return (
    <Suspense fallback={<p role="status">Loading SC-NeuroCore Studio…</p>}>
      <StandaloneStudioPanel />
    </Suspense>
  );
}
