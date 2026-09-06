<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Workspaces, revisions and recovery

A Studio workspace is an append-only list of revisions. Saving adds one; it
never rewrites one. Schema `studio.workspace.v1`.

Three failures this replaces, each reproduced against the previous
single-file store before the replacement was written:

| Failure | What happened | What happens now |
|---|---|---|
| Lost update | Two editors loaded the same workspace and both saved. The second save replaced the first, with nothing to notice it by | A save states the revision it was made from; a save from a stale revision is refused with HTTP 409 |
| Unrecoverable delete | `delete_project` removed the file | Deletion moves the workspace to a trash it can be restored from |
| Torn write | A save truncated the live file and wrote into it. Under a real `RLIMIT_FSIZE` refusal a 156-byte workspace became 65,536 bytes of unterminated JSON, and loading it raised | A revision is written to a sibling temporary file, `fsync`ed and `os.replace`d into place; a failure leaves the previous revision untouched |

## On disk

```
<projects root>/
  <name>/
    head.json          which revision is current
    revisions/1.json   immutable
    revisions/2.json   immutable
  .trash/
    <name>.<ms>/       a deleted workspace, restorable
```

`os.replace` is atomic on POSIX: a reader sees the old bytes or the new ones,
never half of either. The directory is `fsync`ed after the rename, so the
rename itself survives a power loss.

A revision document is also a valid project payload — it carries `name`,
`saved_at`, `version` and `state` — so the evidence bundle reads a revision
directly rather than a second copy that could disagree with it.

## Saving against a revision

`POST /api/project/save` takes `expected_revision`: the revision the caller
loaded and edited.

- **A number** — save on top of that revision. If the workspace has moved on,
  the save is refused.
- **Omitted or `null`** — a claim that the workspace is new. Over an existing
  workspace this is also a conflict, because the caller did not know it was
  there.

A refusal is HTTP 409 with the revision that is actually current:

```json
{
  "detail": {
    "actual_revision": 2,
    "error": "workspace_conflict",
    "expected_revision": 1,
    "reason": "the workspace moved to revision 2 while you were editing revision 1; reload and reapply your change."
  }
}
```

The conflict is translated once, in the Studio API error boundary, because
`WorkspaceConflict` is a `ValueError`: a route that forgot it would report
"invalid input" for a save that was perfectly valid and merely arrived second.

The Studio frontend carries the revision it loaded or last wrote and sends it
with every save. It does **not** adopt the revision a conflict reports —
saving again with the state it still holds is exactly the lost update the
conflict prevents. Reload, reapply, save.

## History, forks and transfer

| Route | What it does |
|---|---|
| `GET /api/project/{name}/revisions` | Every revision, oldest first, with its parent and state digest |
| `GET /api/project/load/{name}?revision=N` | One revision as it was written; no later save can have altered it |
| `POST /api/project/{name}/fork` (`new_name`, optional `revision`) | Copies one revision into a new workspace at revision 1, leaving the source alone |
| `GET /api/project/{name}/export` | A self-contained transfer document |
| `POST /api/project/import` (`name`, `document`) | Creates a workspace from one |

Forking or importing onto a name that is already in use is refused, not
merged.

## Deleting and restoring

`DELETE /api/project/{name}` moves the workspace, with its whole history, into
the trash and returns `recoverable: true`. `GET /api/project/deleted` lists
what is waiting, newest first, each with a `token`; `POST /api/project/restore`
takes that token and brings the workspace back under its original name with
every revision it had.

Restoring onto a name that is live again is refused. Overwriting a live
workspace is the loss this store exists to prevent, and a refusal leaves both
the live workspace and the trashed one intact.

The Studio projects panel shows the trash under **Deleted (restorable)** with a
restore action per entry, so recovery does not require an API call.

Nothing leaves the trash on its own. Retention is an operator decision: remove
a trashed directory to purge it, and it is gone.

## Reading a stored workspace

A workspace written by a **newer** schema is refused rather than downgraded —
dropping fields a future build added would lose a user's work quietly. Upgrade
the package instead.

## Workspaces saved before revisions existed

The previous store kept one flat `<name>.json` per workspace in the project
root. Those files are adopted on first access: the document becomes revision 1
(marked `adopted_from: studio.project-save.v0`) and the flat file is **left on
disk untouched** — an adoption that writes is reversible by deleting the
workspace directory, one that deleted would not be. Adoption is idempotent and
happens on read, list, save and history alike, so nothing an existing
installation saved goes missing.

The first save after an adoption states revision 1, like any other save. Saving
without stating it is refused, as it is for any existing workspace.

A file in the project root that is not a readable workspace — a stray
`notes.txt`, a truncated JSON file — is adopted by nobody, left exactly where it
is, and never allowed to take down the listing.

A revision file that cannot be read is omitted from the history listing rather
than reported as an empty workspace, and every error message about a stored
revision is path-free.
