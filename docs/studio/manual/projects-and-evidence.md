<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SNN Studio user manual -->

# Projects, recovery and evidence

## Choose what to save

| Item | Where it belongs | What it preserves | What you still need |
|---|---|---|---|
| **Project** | Studio server's configured project directory | Workspace settings and revision history | Raw run/evidence exports, dataset and job custody as needed |
| **Session** | This browser profile and origin | Named experiment snapshot | A server copy or export for independent backup |
| **Share link** | URL you copy | The subset of settings encoded by that link | Project/replay pack for a complete handoff |
| Model **Copy link** | URL you copy | Model identity only | All run settings and result |
| JSON / CSV | Your download destination | Returned result / raw trace samples when available | Model, environment and reproducibility contract |
| Replay pack | Your download destination | Sealed request, identity and expected run outcome | A compatible installed runtime to replay it |
| Evidence bundle | Your download destination and/or retained job artifacts | Declared artifacts, receipts and dependency information | Independent scientific and physical qualification for claims beyond its scope |

A project save is not a full backup of datasets, training tensors and every
historical job artifact. A browser session is not a server project.

## Save and reopen a workspace

1. In **Projects**, choose **Save**, enter a new name and wait for acknowledgement.
2. **Refresh** the list and open the saved project.
3. Check its source, parameters, initial state, numerical inputs and graph.
4. Run the restored experiment if you need a fresh result.
5. Save again to append a revision rather than rewriting the previous one.

A save is based on the revision you opened. If another editor saved first,
Studio refuses the stale save with a conflict instead of overwriting that
work. Preserve your draft, compare the latest revision, and deliberately merge
or fork using the [workspace recovery procedure](../workspaces.md#keeping-the-edit-that-was-refused).
Do not blindly retry a stale revision or invent a new expected revision number.

Settings saved with a fresh trial still request a new random draw when run.
Preserve the original JSON and explicitly pin its effective seed with replay
before exporting a pack for that draw. Older workspace
schemas can lack fields that their version never saved; inspect restored
defaults rather than assuming the current editor's values were retained.

## Delete and restore

Deleting a project moves it and its revisions into recoverable trash.
Open **Deleted (restorable)** in the project panel and restore the desired
entry. A live project with the original name prevents restoration onto it;
choose a deliberate resolution that preserves both copies.

Trash is not independent backup. Operator purging removes the retained data.
Browser **Sessions** have a different delete path and do not gain workspace
trash or revision history.

## Browser storage failures

Wait for the status beside **Sessions** after a save or delete. If local storage
is unavailable or full, changes can remain usable in the page without becoming
durable. Reloading can lose a new session or bring back a deletion that could
not be saved. Save a server project and export important results before
reloading. Do not clear all browser storage as the first recovery step.

An origin includes scheme, host and port. `localhost`, `127.0.0.1`, another
port, another browser profile and the native desktop profile can have different
session lists. Confirm the origin and configured project root first.

## Choose an export

| Export | Use it for | Interpretation |
|---|---|---|
| **JSON** | Archiving the returned simulation | Includes metadata, snapshots, raw-data availability and display projection |
| **CSV** | Numeric work in Python/R/MATLAB | Full-resolution trace samples where raw export is admitted; inspect clock and units |
| **SVG** | A figure or presentation | Visual representation, potentially reduced; keep numeric data separately |
| **Generate code** / **Code** view | Running the resolved experiment in Python | The generated standalone script checks the experiment digest before execution |
| **Notebook** | Interactive replay/exploration | Read its recorded request, runner and comparison cells |
| **Replay pack** | Verifying a particular run in another runtime | Compares the recorded expectation under explicit replay rules |
| Scoped project/compile/synthesis bundle | Reviewing the artifacts for that operation | Keep manifest, artifacts and receipts together |
| Evidence cart export | Collecting selected simulation and analysis outcomes | Only queued items and their retained payloads are included |

Open the saved file to check it exists and is readable. A queued item or a
browser download acknowledgement does not independently prove disk persistence.
If a bundle artifact download fails its declared size/SHA-256 check, no verified
file is offered; keep the error and retry the same export context.

Changing an experiment withdraws dependent completion. Historical downloads
remain historical; downloading an older synthesis artifact does not qualify the
current design. Admin exports do not complete an experiment's evidence step.

## Replay a run

**Replay pack** submits the current controls to the backend and executes the
export experiment. It does not simply wrap the last displayed response. For
stochastic runs, preserve the original JSON; copy its effective seed into the
controls and select **replay** before exporting. If **fresh** remains selected,
the export can be a different draw. Check the pack's request and expectation
against the result you intend to hand over. Code/notebook generation also uses
the current export request rather than an arbitrary historical plot.

Use the activated environment and the actual downloaded filename:

```bash
python -m sc_neurocore.studio.replay_pack pack.json --json
```

| Exit | Meaning | Next action |
|---|---|---|
| `0` | Match, or match within the explicitly selected tolerance | Keep the report and its stated scope |
| `1` | Execution completed but the expected result differed | Compare the listed differences and environment |
| `2` | Refused before execution | Resolve the reported identity, runtime or input issue |

Exact comparison is the default. `--tolerance` and `--allow-runtime-drift` change
the admission/comparison contract; do not use them merely to turn a refusal
into success. Historical packs without state samples have narrower comparison
support. [Replay and export](../replay-and-export.md) defines compatibility.
A digest match does not fill missing state, establish authenticity or replace
an independent scientific validation.

## Review and hand off an experiment

1. Save a workspace revision and note its name/revision.
2. Export the replay pack, raw results and relevant evidence bundle.
3. Include source citation, question, changed inputs, seed, backend/version and
   known recording or metric limitations.
4. In **Review**, comment on the saved revision; reply within that revision.
5. Ask the reviewer to check/replay the exported experiment and record differences.

Comments bind to a revision and state digest. A `changed` or `missing` revision
status means the comment cannot qualify the current saved state. See
[Collaboration](../collaboration.md) for author/reviewer responsibilities and
[Evidence verification](../evidence-verification.md) for checking a received
pack. A SHA-256 receipt checks integrity against its manifest; it is not a
signature proving who produced the pack.

## Back up and restore

Inspect the durable-state plan before choosing what to copy:

```bash
sc-neurocore studio-backup-plan --include-local-paths
```

This displays local paths: keep the output private. The command describes a
backup plan; it does not create a backup archive.

Stop new work and coordinate active jobs with the operator before restoring.
Back up the complete configured workspace tree, including `.locks/`, revisions
and trash, and the separately listed durable job/artifact, audit and identity
state that the plan requires. Export browser sessions separately through
saved work/results; copying the server directory does not capture browser
local storage. Protect identity files and do not put them in shared evidence.

Restore into a deliberate directory, select it before launch and check project
names, revisions and retained-job status. Replay a known exported experiment to
check the restored runtime. A desktop application uninstall is designed to
retain user data, but uninstall or same-family rollback is not a substitute for
a verified backup across versions. Read [Workspaces](../workspaces.md) and
[Job ledger](../job-ledger.md#retention-and-backup) for exact recovery boundaries.

Next: [Hardware workflow](hardware.md).
