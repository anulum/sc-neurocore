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
| Pixi / Conda | Every `pixi.lock` | Locked installation, Syft package inventory bound to every artifact, then Grype CPE/NVD matching |

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
remain refused. The native manifest regression resolves the project of the
maintained network lane with the installed Julia 1.13 runtime and compares every generated
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
the Python 3.11 NumPy branch and the platform-specific pytest dependencies. Regenerate it with:

```bash
uv pip compile --universal --python-version 3.11 --generate-hashes \
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

### Conda inventory and vulnerability matching

The Pixi lane installs its committed Mojo environment with `pixi install --locked`.
Syft 1.54.1 catalogs its installed `conda-meta/*.json` records. Each URL, SHA-256,
name, version and build must match the lock, with no omitted or additional package.
Metadata and lock bytes are checked again after scanning. The current adapter
requires one default `linux-64` environment and Conda artifact records; additional
platforms, environments or PyPI records fail until they have a maintained adapter.

Grype 0.120.1 reads that exact Syft SBOM and uses the public Anchore vulnerability
database. The Linux scanner archives are SHA-256 pinned in
`tools/install_conda_audit_scanners.sh`. The database must have a valid hash,
be no older than five days and successfully check for updates. Findings of every
severity remain blocking. Filters, ignored matches, VEX suppression and missing
report settings fail validation. Native commands retain stdout, stderr, exit
status and process-reaping receipts. Database caches remain outside the uploaded
packet; set `GRYPE_DB_CACHE_DIR` to a reusable owned cache directory.

Basilisk was replaced because its audit API began requiring authentication and
our account could not obtain a CI token without a separately granted service
role. Syft and Grype do not need that account or token. This changes the matching
basis: Conda packages use generated CPE identities and NVD data. Such matching
can miss vulnerabilities or misattribute package names and version ranges.
Complete installed-package inventory does not establish complete advisory
coverage. See [Anchore's ecosystem documentation](https://oss.anchore.com/docs/guides/vulnerability/ecosystems/).

### Exact fixed-code exceptions

Each record in `tools/security_scan/advisory_exceptions.toml` names one advisory,
locked package version, artifact digest, fixed source path, source digest and
review-by date.
An exception applies only if its lock exists in the committed inventory, its
exact package and advisory match, and every audit re-establishes its proof.
Unused, expired or unavailable proofs block the audit; no finding is hidden from
the raw Grype output or the accepted/blocked decision.

No record is currently needed. The last one addressed Grype's generic comparison
of Python 3.15.0 against its prerelease fix boundary 3.15.0a6 for CVE-2025-15367:
Python version ordering places the final release after that boundary, and the
installed `poplib.py` had to be byte-identical to its SHA-256-bound file from the
immutable CPython 3.15.0 release commit, which contains the
[vendor fix](https://github.com/python/cpython/pull/143924). On 2026-10-29 the
Mojo lock moved to Python 3.15.2, which also fixes CVE-2026-114583
(`tarfile.tar_filter` link bypass, blocked in 3.15.0). Grype does not report
CVE-2025-15367 for 3.15.2, so the record became unused and was removed. Its
acceptance and refusal rules stay tested on the recorded hosted Grype match
(`tests/fixtures/grype_conda_python_3_15_0_cve_2025_15367.json`) and the
immutable vendor source.

The additional security-packet OSV runner also treats every nonzero scanner
exit as failure. A partially written report cannot turn a service error green.
Audit evidence does not itself establish a release, hosted CI success or
eligibility to delete failed Actions history.
