// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — real Module Federation host harness configuration

import { federation } from "@module-federation/vite";
import react from "@vitejs/plugin-react";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root,
  plugins: [
    react(),
    federation({
      name: "sc_neurocore_contract_host",
      dts: false,
      remotes: {},
      shared: {
        react: { singleton: true, requiredVersion: "19.3.1" },
        "react-dom": { singleton: true, requiredVersion: "19.3.1" },
      },
    }),
  ],
  build: {
    target: "esnext",
    rollupOptions: {
      input: { index: resolve(root, "index.html"), opaque: resolve(root, "opaque.html") },
    },
  },
  server: {
    watch: null,
    host: "127.0.0.1",
    port: 5185,
    strictPort: true,
  },
});
