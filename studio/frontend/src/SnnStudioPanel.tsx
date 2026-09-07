// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — federated Studio panel entry

import App from "./App";
import "./index.css";

/**
 * The Studio's entry component, for standalone and federated hosts alike.
 *
 * It exists so the federation boundary has one stable export to name, and it
 * carries the stylesheet import so a federated host gets the Studio's styles
 * by mounting it rather than by knowing to load them.
 *
 * @returns The Studio shell.
 */
export default function SnnStudioPanel() {
  return <App />;
}
