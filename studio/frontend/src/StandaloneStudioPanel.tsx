// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — standalone Studio application and local editor

import { loader } from "@monaco-editor/react";
import App from "./App";

const editorPath = `${import.meta.env.BASE_URL}vs`;
loader.config({ paths: { vs: new URL(editorPath, import.meta.url).href } });

/**
 * Mount the backend workspace and configure its packaged editor assets.
 *
 * @returns The standalone Studio workspace.
 */
export default function StandaloneStudioPanel() {
  return <App />;
}
