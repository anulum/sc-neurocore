// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Event training input editor

import { useEffect, useRef, useState } from "react";
import { readEventTrainingData } from "../studioEventTrainingData";
import type { TrainingConfigurationProps } from "./TrainingConfiguration";

/** Event declaration controls that keep drafts separate from applied configuration. */
interface TrainingEventInputProps {
  config: TrainingConfigurationProps["config"];
  setConfig: TrainingConfigurationProps["setConfig"];
  onDirtyChange: (dirty: boolean) => void;
}

/**
 * Import and apply the manifest, split and encoder as one portable declaration.
 *
 * @param props - Current configuration, setter and draft readiness callback.
 * @returns File/paste import, diagnostics and the applied input's identity.
 */
export default function TrainingEventInput({ config, setConfig, onDirtyChange }: TrainingEventInputProps) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const version = useRef(0);
  useEffect(() => {
    version.current += 1;
    setText(config.event_data === undefined ? "" : JSON.stringify(config.event_data, null, 2));
    setError(null);
    onDirtyChange(false);
    return () => { version.current += 1; };
  }, [config.event_data, onDirtyChange]);

  /**
   * Apply only a matching envelope, retaining the previous configuration on error.
   */
  function apply() {
    try {
      const value: unknown = JSON.parse(text);
      if (typeof value !== "object" || value === null || !("encoder" in value)
        || typeof value.encoder !== "object" || value.encoder === null || !("n_steps" in value.encoder)
        || typeof value.encoder.n_steps !== "number" || !Number.isInteger(value.encoder.n_steps)
        || value.encoder.n_steps < 1) {
        throw new Error("Event input needs a positive integer encoder window");
      }
      const input = readEventTrainingData(value, config.dataset, value.encoder.n_steps);
      setConfig("timesteps", value.encoder.n_steps);
      setConfig("event_data", input);
      setError(null);
      onDirtyChange(false);
    } catch (failure: unknown) {
      setError(failure instanceof Error ? failure.message : "Event input could not be read");
      onDirtyChange(true);
    }
  }

  /**
   * Load a selected file as a draft; a later edit or selection supersedes this read.
   *
   * @param file - Local declaration file, never sent as a filesystem path.
   */
  async function loadFile(file: File) {
    const selected = ++version.current;
    onDirtyChange(true);
    try {
      const contents = await file.text();
      if (selected !== version.current) return;
      setText(contents);
      setError(null);
    } catch (failure: unknown) {
      if (selected === version.current) {
        setError(failure instanceof Error ? failure.message : "Event input file could not be read");
      }
    }
  }

  return <section aria-label="Event training input" style={{ padding: 12, borderBottom: "1px solid var(--border)" }}>
    <p>Import the event manifest, group split and encoder declaration. Dataset files must be available to the server operator.</p>
    <label>
      Event input file
      <input aria-label="Event input file" type="file" accept="application/json,.json" onChange={(event) => {
        const file = event.target.files?.[0];
        if (file !== undefined) void loadFile(file);
        event.target.value = "";
      }} />
    </label>
    <label style={{ display: "block" }}>
      Event input JSON
      <textarea aria-label="Event input JSON" value={text} rows={6} style={{ display: "block", width: "100%", fontFamily: "monospace" }}
        onChange={(event) => {
          version.current += 1;
          setText(event.target.value);
          setError(null);
          onDirtyChange(true);
        }} />
    </label>
    <button type="button" onClick={apply}>Apply event input</button>
    <button type="button" onClick={() => {
      version.current += 1;
      setText("");
      setError(null);
      setConfig("event_data", undefined);
      onDirtyChange(false);
    }}>Clear event input</button>
    {error !== null && <p role="alert">{error}</p>}
    {config.event_data !== undefined && <dl style={{ overflowWrap: "anywhere" }}>
      <dt>Applied dataset</dt><dd>{config.dataset}</dd>
      <dt>Window</dt><dd>{String(config.event_data.encoder.n_steps)} steps × {String(config.event_data.encoder.dt_ms)} ms</dd>
      <dt>Split</dt><dd>{config.event_data.train_split} / {config.event_data.evaluation_split}</dd>
      <dt>Manifest digest</dt><dd>{config.event_data.digests.manifest}</dd>
      <dt>Encoder digest</dt><dd>{config.event_data.digests.encoder}</dd>
    </dl>}
    <p>The server verifies declarations, digests and recordings before admitting training.</p>
  </section>;
}
