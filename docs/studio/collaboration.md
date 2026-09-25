<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Reviewing and Sharing Experiments

This page is the working procedure for two researchers who review each other's
Studio experiments: what the author hands over, how the reviewer checks it,
where the review is written down, and what to do when both edited the same
thing. Each step points to the page that specifies it in full.

The collaboration is asynchronous and revision-based. There is no live
co-editing and no automatic merge: a comment is bound to one saved revision,
and two diverging edits are kept as two workspaces.

## What the author hands over

Save the experiment first. Every save is an immutable, numbered revision with
a SHA-256 digest of its state (`state_sha256`), listed by
`GET /api/project/{name}/revisions` ([Workspaces](workspaces.md#history-forks-and-transfer)).
A review refers to a revision number and that digest, so an unsaved edit cannot
be reviewed.

Then choose what to send, by what the reviewer must be able to do:

| The reviewer should… | Send | Made by |
|---|---|---|
| open and edit the same Studio state | a workspace transfer document | `GET /api/project/{name}/export` (optionally `?revision=N`) |
| check that the run reproduces on their machine | a replay pack | **Replay pack**, `POST /api/export/replay-pack` ([Replay & Export](replay-and-export.md#the-replay-pack)) |
| read and rerun the experiment outside the Studio | a replay notebook | **Notebook**, `POST /api/export/replay-notebook` ([Replay & Export](replay-and-export.md#notebook)) |
| rebuild a network with the public API | a tutorial notebook | **Notebook** on the canvas, `POST /api/graph/notebook` ([Network Canvas](network-canvas.md#tutorial-notebook)) |

None of these carries a path from the author's machine. A transfer document
holds one revision's state, not the workspace history and not its review
comments. Send the pack with the transfer document when both are needed: the
document says what was set up, and the pack says what that setup produced.

## What the reviewer checks

1. **That the run reproduces.**
   - Replay the pack with the installed package:
     `python -m sc_neurocore.studio.replay_pack pack.json`.
   - Exit code 0 is a reproduction and 1 is a run that differs, with every
     difference listed. Exit code 2 is a refusal before anything ran, with
     the reason.
   - Another installation normally differs in package, interpreter or NumPy
     version. `--allow-runtime-drift` runs the pack anyway and reports the
     differences, instead of refusing.
   - A `match-within-tolerance` verdict is only as strong as the `--tolerance`
     given. State the tolerance in the review.
2. **That it is the same experiment.**
   - Import the transfer document with `POST /api/project/import`. It becomes
     revision 1 of a new workspace on the reviewer's side.
   - Its `state_sha256` equals the author's revision digest when the state
     arrived unchanged. Compare the two digests before commenting: equal
     digests mean both sides are discussing the same state.
3. **What the model is.**
   - The first cell of a replay notebook cites the catalogue model from its
     descriptor, or says the equations have no published source.
   - The model's readiness label ([Model Readiness](model-readiness.md))
     states what evidence exists for it.
   - The notebooks run the software model only. No fixed-point, RTL, synthesis
     or board step is part of them, and each notebook says so.

## Writing the review

Comment on the revision itself, in the **Review** tab or with
`POST /api/project/{name}/revisions/{revision}/comments`
([Workspaces](workspaces.md#reviewing-a-revision)).

- **What a comment records:** the revision number, the digest of that
  revision's state, and the author. The author is the authenticated principal,
  or `local` in the single-user lab profile.
- **Replies:** a reply must answer a comment on the same revision, so a thread
  never mixes two states.
- **Revisions that change:** comments are appended and never rewritten. When
  comments are listed, each revision is checked again:
  - `revision_status` is `matches` while the revision still has the digest the
    comment was written against;
  - it is `changed` once that digest no longer holds;
  - it is `missing` once the revision is gone.

  A comment is therefore never shown as if it applied to a different state.

Comments live on the machine where they were written. When the reviewer works
on an imported copy, the comments are on the reviewer's workspace. Send them
back with the revision number and `state_sha256` they are bound to, so that the
author can match them to their own revision.

A useful review names the revision and digest, gives the replay command and its
exit code, the verdict and tolerance, and any runtime drift reported. Each
objection should point to a block of the experiment (model, parameters,
protocol, randomness, backend) rather than to the experiment as a whole.

## When both edited the same thing

A save that arrives second is refused with the revision that is current
([Workspaces](workspaces.md#keeping-the-edit-that-was-refused)). Keep the
refused edit with `POST /api/project/{name}/branch-refused-edit`. It becomes
revision 1 of its own workspace, named for the revision it diverged from, and
both edits stay loadable, exportable and comparable. The Studio does not merge
them, because combining two experiment states is a scientific decision. Decide
which parameters stand, save that as a new revision, and review it.

To try a variant without touching the author's workspace, fork a revision with
`POST /api/project/{name}/fork`.

## What this does not provide

- No realtime co-editing, presence or locking beyond the refused second save.
- No transfer of workspace history or review comments between machines. The
  transfer document is one revision.
- No hardware claim from a shared notebook or pack. Execution on a target is a
  separate, opt-in step owned by whoever operates the device.

For contributing code, models or documentation to SC-NeuroCore itself, see
[CONTRIBUTING.md](https://github.com/anulum/sc-neurocore/blob/main/CONTRIBUTING.md).
