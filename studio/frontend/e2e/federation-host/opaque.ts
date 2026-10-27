// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — opaque producer transport fixture; not the SCPN Studio Hub

const parameters = new URLSearchParams(location.search);
const wire = parameters.get("catalogue");
if (wire === null) throw new Error("Opaque producer fixture requires a catalogue.");
const catalogue: unknown = JSON.parse(wire);
let refuse = parameters.get("reply") === "refuse";
const panelRoot = document.getElementById("panel-root");
const requests = document.getElementById("bridge-requests");
const allow = document.getElementById("allow-catalogue");
if (panelRoot === null || requests === null || !(allow instanceof HTMLButtonElement)) {
  throw new Error("Opaque producer fixture controls are missing.");
}
const panel = document.createElement("iframe");
panel.title = "Opaque producer panel";
panel.setAttribute("sandbox", "allow-scripts");
panel.src = "./index.html";
panel.width = "100%";
panel.height = "800";
allow.disabled = !refuse;
allow.addEventListener("click", () => { refuse = false; allow.disabled = true; });
window.addEventListener("message", (event: MessageEvent<unknown>) => {
  const data = event.data;
  if (event.source !== panel.contentWindow || typeof data !== "object" || data === null
    || !("type" in data) || data.type !== "scpn:bridge:request"
    || !("id" in data) || typeof data.id !== "string"
    || !("capability" in data) || typeof data.capability !== "string") return;
  const item = document.createElement("li");
  item.dataset.id = data.id;
  item.dataset.capability = data.capability;
  item.dataset.origin = event.origin;
  item.textContent = data.capability;
  requests.append(item);
  const ok = !refuse && data.capability === "federation:config";
  panel.contentWindow?.postMessage({ type: "scpn:bridge:response", id: data.id, ok,
    ...(ok ? { result: catalogue } : { error: "Private fixture diagnostic" }),
  }, "*");
});
panelRoot.append(panel);
