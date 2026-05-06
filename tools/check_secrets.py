# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pre-commit secret scanner

"""Scan tracked files for leaked secrets, API keys, and credentials.

Usage::

    python tools/check_secrets.py          # scan repo
    python tools/check_secrets.py --fix    # scan + suggest .gitignore

Checks for:
    - API keys (Bearer, token=, api_key=, sk-...)
    - Private keys (-----BEGIN ... PRIVATE KEY-----)
    - .env file references that might contain secrets
    - Password patterns (password=, passwd=, secret=)
    - AWS/GCP/Azure credentials
    - Hardcoded URLs with embedded credentials

Allowed patterns (not flagged):
    - ``agentic-shared`` local path reference (it's a local path, not a credential)
    - ``api_key="local"`` (dummy key for local llama-server)
    - ``LOCAL_LLM_API_KEY=local`` (dummy for local endpoint)

Exit codes:
    0 — clean (no secrets found)
    1 — secrets detected
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


# Patterns that indicate a potential secret leak
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "API key",
        re.compile(
            r'(?:api[_-]?key|apikey|token)\s*[=:]\s*["\']?[A-Za-z0-9_\-]{20,}', re.IGNORECASE
        ),
    ),
    ("Bearer token", re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE)),
    ("sk-prefixed API key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("AWS key", re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    ("Private key", re.compile(r"-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH)?\s*PRIVATE KEY-----")),
    (
        "Password",
        re.compile(r'(?:password|passwd|secret)\s*[=:]\s*["\'][^"\']{4,}["\']', re.IGNORECASE),
    ),
    ("GCP service account", re.compile(r'"type"\s*:\s*"service_account"')),
    ("Hardcoded cred URL", re.compile(r"https?://[^@\s]+:[^@\s]+@[^/\s]+")),
    (".env file", re.compile(r"\.env\b(?!\.example|\.template|\.dist)", re.IGNORECASE)),
]

# Patterns that are explicitly allowed (false positives)
_ALLOWED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"agentic-shared"),  # local path, not a secret
    re.compile(r'api_key\s*=\s*["\']?local'),  # dummy key for llama-server
    re.compile(r"LOCAL_LLM_API_KEY\s*=\s*local"),  # dummy env var
    re.compile(r"Bearer\s+local"),  # dummy bearer
    re.compile(r"\.env\.\w+"),  # .env.example, .env.template
    re.compile(r"#.*\.env"),  # commented out
    re.compile(r"check_secrets"),  # self-reference
]

# File extensions to skip (binary, media, etc.)
_SKIP_EXTENSIONS = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".webp",
        ".mp4",
        ".avi",
        ".mov",
        ".webm",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".whl",
        ".egg",
        ".gguf",  # LLM model files
    }
)


def _get_tracked_files(repo_root: Path) -> list[Path]:
    """Get list of git-tracked files."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return [repo_root / f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Fallback: walk the tree
    files = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d != ".git" and not d.startswith("__pycache__")]
        for fname in filenames:
            files.append(Path(dirpath) / fname)
    return files


def _is_allowed(line: str) -> bool:
    """Check if a line matches any allowed pattern."""
    return any(p.search(line) for p in _ALLOWED_PATTERNS)


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string in bits."""
    if not s:
        return 0.0
    from collections import Counter
    import math

    freq = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _find_high_entropy_strings(
    line: str, threshold: float = 4.5, min_length: int = 16
) -> list[str]:
    """Find substrings with entropy above threshold (likely secrets)."""
    # Split on common delimiters
    candidates = re.findall(r"[A-Za-z0-9+/=_\-]{16,}", line)
    return [c for c in candidates if len(c) >= min_length and _shannon_entropy(c) >= threshold]


def _scan_git_history(repo_root: Path, max_commits: int = 100) -> list[dict[str, str]]:
    """Scan recent git history for secrets that were committed then removed."""
    findings: list[dict[str, str]] = []
    try:
        result = subprocess.run(
            ["git", "log", f"-{max_commits}", "--diff-filter=D", "--name-only", "--format=%H"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return findings

        # Check deleted files for secret patterns
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Check if any deleted file had a suspicious name
            lower = line.lower()
            if any(
                kw in lower for kw in [".env", "credentials", "secret", ".pem", ".key", "token"]
            ):
                findings.append(
                    {
                        "file": f"[DELETED] {line}",
                        "line_num": "git-history",
                        "pattern": "Deleted sensitive file",
                    }
                )
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return findings


def scan_repo(repo_root: Path, scan_history: bool = True) -> list[dict[str, str]]:
    """Scan repository for secret leaks.

    Performs three layers of detection:
    1. Regex pattern matching (API keys, tokens, passwords)
    2. Shannon entropy analysis (high-entropy strings > 4.5 bits)
    3. Git history scan (deleted sensitive files)

    Returns
    -------
    list[dict]
        Each dict: ``{"file", "line_num", "pattern"}``.
    """
    findings: list[dict[str, str]] = []
    files = _get_tracked_files(repo_root)

    for fpath in files:
        if fpath.suffix.lower() in _SKIP_EXTENSIONS:
            continue
        if not fpath.is_file():
            continue
        try:
            stat = fpath.stat()
            if stat.st_size > 512 * 1024:
                continue
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue

        rel_path = str(fpath.relative_to(repo_root))
        if fpath.name == "check_secrets.py":
            continue
        for line_num, line in enumerate(text.split("\n"), 1):
            if _is_allowed(line):
                continue

            # Layer 1: Regex patterns
            for pattern_name, pattern in _SECRET_PATTERNS:
                match = pattern.search(line)
                if match:
                    findings.append(
                        {
                            "file": rel_path,
                            "line_num": str(line_num),
                            "pattern": pattern_name,
                        }
                    )
                    break

            # Layer 2: Entropy analysis
            high_entropy = _find_high_entropy_strings(line)
            for he_str in high_entropy:
                entropy = _shannon_entropy(he_str)
                findings.append(
                    {
                        "file": rel_path,
                        "line_num": str(line_num),
                        "pattern": f"High entropy ({entropy:.1f} bits)",
                    }
                )

    # Layer 3: Git history
    if scan_history:
        findings.extend(_scan_git_history(repo_root))

    return findings


def main() -> int:
    """CLI entry point."""
    repo_root = Path(__file__).resolve().parent.parent
    # Ensure we're in the repo root
    if not (repo_root / "pyproject.toml").exists():
        print(f"ERROR: Not a repo root: {repo_root}", file=sys.stderr)
        return 2

    print(f"Scanning {repo_root} for secrets...")
    findings = scan_repo(repo_root)

    if not findings:
        print("\033[32m✓ No secrets found. Repository is clean.\033[0m")
        return 0

    print(f"\n\033[31m✗ Found {len(findings)} potential secret(s):\033[0m\n")
    for f in findings:
        print(f"  {f['file']}:{f['line_num']}")
        print("    Finding details redacted.")
        print()

    return 1


if __name__ == "__main__":
    sys.exit(main())
