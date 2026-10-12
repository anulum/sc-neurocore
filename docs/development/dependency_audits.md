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
| Julia | Every format-2.0 or format-2.1 `Manifest.toml` | OSV queries for every exact package version, including build suffixes |
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

Julia 1.13 writes format 2.1 manifests with registry metadata. Both supported
formats retain every dependency entry; path, Git and unversioned dependencies
remain refused. The native manifest regression resolves the maintained FFTW
project with the installed Julia 1.13 runtime and compares every generated
package/version identity with the audit queries.

The Julia network lane depends on the `FFTW_jll` binary and calls it directly.
Depending on `FFTW.jl` instead adds the package manager and the downloader to
the lock through the optional MKL provider, together with the curl, Git, SSH
and TLS libraries of the Julia distribution. The lane test refuses those
entries. The Julia distribution itself still ships these libraries; a manifest
records the packages a project loads, not the runtime that loads them.

npm silently omits unversioned packages from advisory submission. Before running
it, the adapter validates every installation entry in a version-2 or version-3
lock, retaining its path, registry name and exact semantic version. Linked,
unversioned, nonregistry and empty dependency inventories fail. These require a
reviewed advisory adapter before they can enter a passing audit. The response
must cover the complete validated package count, with integer counters for all
severities. A zero exit status or a clean-looking total alone is insufficient.

The Studio lock selects `source-map-js` 1.2.2 for
[GHSA-68fv-2mgg-jv7q](https://github.com/advisories/GHSA-68fv-2mgg-jv7q).
The update retains the remaining dependency graph. Its npm audit still includes
development, optional and peer dependencies; a passing npm result does not
qualify the other ecosystem jobs or the whole security packet.

The installed native-wheel profile is generated from `requirements/wheels-test.in`
with the development lock as its constraint. Its universal resolution retains
the Python 3.10 and 3.11 NumPy branches and the platform-specific pytest
dependencies. Regenerate it with:

```bash
uv pip compile --universal --python-version 3.10 --generate-hashes \
  --constraint requirements/ci-dev.txt \
  --output-file requirements/wheels-test.txt requirements/wheels-test.in
```

This profile includes the framework and shared-fixture dependency used by the
installed batch-encoding contracts. It remains part of the blocking Python
lock audit, including every marker branch.

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

The pinned Pixi audit 0.2.0 covers conda-forge and PyPI. Its `matches`, `ignored`,
`not_audited` and `unused_ignores` lists must agree with typed counters and cover
every locked package. A package reported as `not_audited` blocks the lock
unless a vendor-record exception for exactly that package version holds. A Pixi
lock without a
readable package record list is refused before the scanner starts. No blanket waiver
or automatic `--fix` is applied. Advisory remediation needs reviewed fixed
versions and ordinary integration validation before changing the lock.

### Advisory exceptions

A Pixi finding can be accepted only through one table of
`tools/security_scan/advisory_exceptions.toml`, naming the lock, the package
version and the advisory, with one of two reasons that the audit re-establishes
on every run. A package the scanner cannot audit has one table without an
advisory and the third reason:

- `no_fixed_version_in_channel` — the locked artefact is the newest version
  and the newest build its channel publishes for its platform. Channel and
  platform are read from the lock, not from the table. The check reads the
  channel index and requires the locked file to be listed there with the
  lock's digest; a newer version, or a newer build of the same version, fails
  it, so a published fix turns the audit red by itself.
- `component_not_in_package` — the advisory concerns a program the locked
  package does not ship. The check downloads the exact locked artefact,
  verifies it against the lock digest and requires that none of its declared
  paths matches the listed patterns.
- `vendor_records_reviewed` — the scanner has no advisory data for the vendor
  channel, so the audit reads the vendor's records itself. The check reads
  the platform index beside the locked file and requires the file to be
  listed with the lock's digest. It then asks the National Vulnerability
  Database for every record bound to a product identifier of the named
  vendor. The records must be exactly those listed as reviewed, and none may
  name a package of the lock as its product. A new record, a withdrawn
  record and a withdrawn file all fail the check.

There is no severity, confidence or reachability filter and no wildcard. A
finding without a table, a table whose check fails or cannot be completed, a
table that matches no finding, and a table that names another ecosystem or a
lock the audit does not read all fail the audit. Whether the listed paths
describe the affected program is not something a machine can check; it is
reviewed with each table, and the evidence records how many paths the
artefact declares. The vendor check has two limits of the same kind. It sees
only records that already carry a product identifier, and the database
assigns those some time after publication. Whether a reviewed record that
names another product of the vendor also affects a locked package is reviewed
with the table. Each run writes the accepted findings and packages with their
evidence to `exceptions.json` in the audit packet: an accepted finding is a
known, recorded limitation, not an absence of findings, and a package accepted
by its vendor records has not been audited by the scanner.

The additional security-packet OSV runner also treats every nonzero scanner
exit as failure. A partially written report cannot turn a service error green.
Audit evidence does not itself establish a release, hosted CI success or
eligibility to delete failed Actions history.
