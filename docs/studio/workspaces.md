<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Workspaces, revisions and recovery

A Studio workspace is an append-only list of revisions. Saving adds one; it
never rewrites one. Schema `studio.workspace.v1`.

Four failures this replaces, each reproduced against the store as it stood
before the replacement was written:

| Failure | What happened | What happens now |
|---|---|---|
| Lost update | Two editors loaded the same workspace and both saved. The second save replaced the first, with nothing to notice it by | A save states the revision it was made from; a save from a stale revision is refused with HTTP 409 |
| Unrecoverable delete | `delete_project` removed the file | Deletion moves the workspace to a trash it can be restored from |
| Torn write | A save truncated the live file and wrote into it. Under a real `RLIMIT_FSIZE` refusal a 156-byte workspace became 65,536 bytes of unterminated JSON, and loading it raised | A revision is written to a sibling temporary file, `fsync`ed and `os.replace`d into place; a failure leaves the previous revision untouched |
| Lost update through the check | Two savers read the same head between them, both computed revision 1, both were acknowledged with different digests, and one payload was overwritten | Reading the head and writing the revision happen while the workspace is held against every other writer, in this process and in others |

## On disk

```
<projects root>/
  <name>/
    head.json          which revision is current
    revisions/1.json   immutable
    revisions/2.json   immutable
  .trash/
    <name>.<ms>-<random>/   a deleted workspace, restorable
  .locks/
    <name>.lock.sqlite3   holds the workspace while one writer works on it
    <name>.legacy-retired prevents re-adoption of a deleted legacy workspace
```

`os.replace` is atomic on POSIX: a reader sees the old bytes or the new ones,
never half of either. The directory is `fsync`ed after the rename, so the
rename itself survives a power loss.

Trash tokens are opaque; the random suffix keeps two deletions under an equal
clock reading separate. Older `<name>.<ms>` tokens remain readable and restorable.
Deletion of an adopted workspace preserves the original flat JSON file and first
writes a durable retirement marker in `.locks/`. Reads cannot resurrect that
older copy after deletion or restart. Restore uses the saved revision directory,
not the flat migration source. Include `.locks/` in workspace backups. Rolling
back to a version without retirement-marker support can re-adopt deleted legacy
files; restore or archive those migration sources before such a rollback.

A revision document is also a valid project payload — it carries `name`,
`saved_at`, `version` and `state` — so the evidence bundle reads a revision
directly rather than a second copy that could disagree with it.

## One writer at a time

Refusing a stale save is only a refusal if reading the head and writing the
next revision happen together. Every operation that writes a workspace — save,
fork, import, delete, restore, and the adoption of a pre-revision file — holds
that workspace for as long as it needs it, and so does every read a writer
depends on.

The exclusion is a SQLite `BEGIN IMMEDIATE` on a small database of its own, one
per workspace, which is the primitive the job ledger already relies on for
single-host serialisation:

- it is enforced by the operating system, so **two server processes** over one
  project root exclude each other, not only two threads of one process;
- it is released when the connection closes **and when the process dies**, so a
  crashed or killed worker cannot leave a workspace permanently locked;
- the wait is **bounded**. A writer that cannot take the workspace within the
  wait is answered `503` with `error: workspace_busy` and writes nothing, so
  the same request can simply be retried; a request thread is never blocked
  indefinitely by another writer.

The lock databases live in `.locks/` beside the workspaces rather than inside
one, because deleting a workspace moves its directory into the trash: a lock
kept inside would vanish underneath the writer holding it.

A revision number is never reused. A writer that died between writing its
revision file and writing the head leaves a revision the head does not point
at; the next save numbers itself above the highest revision on disk rather than
above the head, so that stored state is never written over. It stays readable
through `GET /api/project/load/{name}?revision=N` and appears in the history.

**Operator limitation, stated rather than implied.** This serialises writers
that share a filesystem implementing SQLite's locking — one host, as for the
job ledger. Two hosts over one network share are outside what it can promise.
Deploy one host per project root.

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

## Keeping the edit that was refused

Refusing the save protects the other editor's work, and on its own it leaves
the refused edit in one browser and nowhere else: "reapply your change" means
retype it, and a closed tab loses it. That is a second way to lose an update,
slower than the first.

`POST /api/project/{name}/branch-refused-edit` stores the refused state as the
first revision of its own workspace, named for what it diverged from —
`column (from revision 1)`. Both edits then exist as revisions and either can
be loaded, exported or compared; nothing merges them automatically, because a
merge of two Studio states is a scientific judgement and not a textual one.

It refuses a `base_revision` that does not exist — an edit that diverged from
nothing is not a divergence — and it never overwrites an existing workspace, so
a second conflict cannot bury the first branch.

A `503` is a different answer and the editor says so differently: nothing was
written, no other editor's work is at stake, and the same save can simply be
sent again. Treating it like a conflict would push a user into reloading and
reapplying work that never conflicted with anything.

```json
{
  "detail": {
    "error": "workspace_busy",
    "name": "shared",
    "reason": "another writer held workspace 'shared' for longer than 10.0 seconds; nothing was written, so the same save can be retried.",
    "timeout_seconds": 10.0
  }
}
```

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

## Reviewing a revision

A comment is made on one saved revision, never on a workspace in general:
`POST /api/project/{name}/revisions/{revision}/comments` with a `body` (1 to
4000 characters) and, for a reply, `reply_to` — a comment on the same
revision. The author is the request's authenticated principal, or `local` in
the single-user lab profile. The comment records the revision number and the
digest of that revision's state, and is appended to `review.jsonl` beside the
revisions; comments are never rewritten.

`GET /api/project/{name}/comments` (optionally `?revision=N`) lists them, each
checked against its revision as it is read: `revision_status` is `matches`,
`changed` when the revision's state no longer has the digest it was reviewed
at, or `missing` when the revision is gone — so a comment is never shown as if
it applied to something else. The **Review** tab shows the comments on the
revision the editor opened or last saved, threads replies under the comment
they answer, and adds comments and replies.

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
