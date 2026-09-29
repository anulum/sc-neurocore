// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — a named, labelled text prompt in place of window.prompt

import { useEffect, useRef, useState } from "react";

/** What to ask for and what to do with the answer. */
export interface TextPromptRequest {
  /** The dialog's heading, which is also its accessible name. */
  title: string;
  /** The field's label. */
  label: string;
  /** The confirming button's label. */
  confirmLabel: string;
  /** A multi-line field instead of a single line. */
  multiline?: boolean;
  /** Called with the trimmed, non-empty answer. */
  onSubmit: (value: string) => void;
}

/**
 * The field and buttons of one prompt; keyed by the caller so every request
 * starts with an empty field.
 *
 * @param props - The request and what to do when the prompt closes.
 * @returns The form.
 */
function TextPromptForm({ request, onClose }: { request: TextPromptRequest; onClose: () => void }) {
  const [value, setValue] = useState("");
  const answer = value.trim();
  return (
    <form
      method="dialog"
      onSubmit={(event) => {
        event.preventDefault();
        if (answer === "") return;
        request.onSubmit(answer);
        onClose();
      }}
    >
      <h2 id="text-prompt-title">{request.title}</h2>
      <label className="text-prompt-field">
        <span>{request.label}</span>
        {request.multiline === true ? (
          <textarea autoFocus rows={8} value={value} onChange={(event) => { setValue(event.target.value); }} />
        ) : (
          <input type="text" autoFocus value={value} onChange={(event) => { setValue(event.target.value); }} />
        )}
      </label>
      <div className="text-prompt-actions">
        <button type="button" className="btn-simulate btn btn--ghost" onClick={onClose}>Cancel</button>
        <button type="submit" className="btn-simulate btn btn--primary" disabled={answer === ""}>
          {request.confirmLabel}
        </button>
      </div>
    </form>
  );
}

/**
 * A modal text prompt.
 *
 * `window.prompt` gave the Studio's save and import actions an unstyled
 * browser box: no labelled field, no multi-line input for a pasted trace, and
 * no way to style or test it as part of the page. This is a native `dialog`
 * opened with `showModal`, so the browser keeps focus inside it and closes it
 * on Escape; the field is labelled and the heading names the dialog.
 *
 * @param props - The pending request (none when closed), a key that changes
 *   with every request, and what to do when the prompt closes.
 * @returns The dialog.
 */
export default function TextPromptDialog({
  request,
  requestKey,
  onClose,
}: {
  request: TextPromptRequest | null;
  requestKey: number;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    if (element === null) return;
    if (request !== null && !element.open) element.showModal();
    if (request === null && element.open) element.close();
  }, [request]);
  return (
    <dialog ref={dialog} className="text-prompt" aria-labelledby="text-prompt-title" onClose={onClose}>
      {request !== null && <TextPromptForm key={requestKey} request={request} onClose={onClose} />}
    </dialog>
  );
}
