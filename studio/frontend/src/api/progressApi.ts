// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: progress endpoints.
import { progressWebSocketProtocols } from "./http";
import type {
  ProgressMessage,
} from "./types";

export function connectProgress(
  op: string,
  config: Record<string, unknown>,
  onMessage: (msg: ProgressMessage) => void,
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(
    `${proto}//${window.location.host}/ws/progress`,
    progressWebSocketProtocols(),
  );
  ws.onopen = () => ws.send(JSON.stringify({ op, config }));
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data));
    } catch { /* ignore parse errors */ }
  };
  ws.onerror = () => onMessage({ type: "error", msg: "WebSocket connection failed" });
  return ws;
}

