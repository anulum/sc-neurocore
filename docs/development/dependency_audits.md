# Committed dependency audits

`Dependency Lock Audit` runs on every push and on every pull request to `main`.
Its `dependency-lock-audit` aggregate succeeds only when all six ecosystem
jobs succeed. Branch protection must require this aggregate after its first
hosted run has reported; a workflow file alone does not establish that setting.

The inventory comes from checked `git ls-files` output. Each job records every
lock digest before and after execution. New locks in a supported format enter
the audit automatically; an unsupported additional `.lock` format fails the
inventory. Project dependencies are neither installed nor rewritten by audits.

| Ecosystem | Committed input | Advisory path |
|---|---|---|
| Python | Every hashlocked `requirements/*.txt` profile | Strict pip-audit queries for every pinned version and marker branch |
| Rust | Every `Cargo.lock` | `cargo audit --file … --format json --deny warnings --url https://github.com/RustSec/advisory-db` |
| npm | Every `package-lock.json` | Exact registry identity validation, then `npm audit --package-lock-only --include=dev --include=optional --include=peer --registry=https://registry.npmjs.org --json` |
| Go | Every `go.mod` and associated `go.sum` | OSV queries for standard library, module pins and checksum-only versions |
| Julia | Every version-2 `Manifest.toml` | OSV queries for every exact package version, including build suffixes |
| Pixi | Every `pixi.lock` | Pinned pixi-audit, with complete JSON coverage validation |

Run an ecosystem without running a test suite:

```bash
python -m tools.security_scan.locked_dependency_audit \
  --ecosystem julia --output-dir /path/to/new-audit-packet
```

Findings of every severity, ignored or unchecked packages, malformed responses,
database errors, tool failures, timeouts and changed input bytes block the job.
The direct OSV adapter retains each request and response page, including HTTP
error bodies. Per-page receipts bind the request and response digests and HTTP
status. Connection failures retain their exception types, without exception
text or headers. Responses remain bounded to 16 MiB plus one detection byte;
each request has a 30-second timeout. Go call analysis does not filter advisory
results. Julia package versions are not converted to Python or generic upstream
versions.

npm silently omits unversioned packages from advisory submission. Before running
it, the adapter validates every installation entry in a version-2 or version-3
lock, retaining its path, registry name and exact semantic version. Linked,
unversioned, nonregistry and empty dependency inventories fail. These require a
reviewed advisory adapter before they can enter a passing audit. The response
must cover the complete validated package count, with integer counters for all
severities. A zero exit status or a clean-looking total alone is insufficient.

Rust report validation requires explicit unfiltered settings, all supported
informational warning categories, consistent typed finding counters and zero
warnings. Missing settings cannot stand in for an unsuppressed audit.
Effective project or Cargo-home audit configuration is digest-bound before and
after scanning. Disabled database fetch, stale acceptance and database origin
or cache overrides fail before the scanner runs. A custom advisory database
requires a separately reviewed custody adapter.

The reviewed Python CPU profile preserves `torch`'s `+cpu` lock identity,
download origin and hashes while querying the upstream public version. Its
report explicitly records both identities. Other local version suffixes are
refused. This checks upstream package advisories; it does not attest that a
particular CPU wheel has no build-specific vulnerability.

The current Pixi beta covers conda-forge and PyPI. Modular packages reported as
unchecked cannot establish a passing complete-lock audit. A Pixi lock without a
readable package record list is refused before the scanner starts. No blanket waiver
or automatic `--fix` is applied. Advisory remediation needs reviewed fixed
versions and ordinary integration validation before changing the lock.

The additional security-packet OSV runner also treats every nonzero scanner
exit as failure. A partially written report cannot turn a service error green.
Audit evidence does not itself establish a release, hosted CI success or
eligibility to delete failed Actions history.
