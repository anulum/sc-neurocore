# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Configuration Validator for SC-NeuroCore
========================================

Checks for the existence and validity of key configuration files:
- pyproject.toml
- HDL files
- Test directory structure
"""

import os
import sys
import tomli # Requires tomli or python 3.11+ tomllib. fallback to simple read.

def validate_project_structure():
    print("Validating Project Structure...")
    required_paths = [
        "pyproject.toml",
        "src/sc_neurocore",
        "hdl",
        "tests",
        "docs/API_REFERENCE.md",
        "docs/USER_GUIDE.md"
    ]
    
    all_ok = True
    for p in required_paths:
        if os.path.exists(p):
            print(f"  [OK] Found {p}")
        else:
            print(f"  [MISSING] {p}")
            all_ok = False
            
    if all_ok:
        print("Structure Validation: PASSED")
    else:
        print("Structure Validation: FAILED")
        sys.exit(1)

def validate_pyproject():
    print("\nValidating pyproject.toml...")
    if not os.path.exists("pyproject.toml"):
        print("  [ERROR] pyproject.toml missing")
        return

    try:
        if sys.version_info >= (3, 11):
            import tomllib as toml
        else:
            try:
                import tomli as toml
            except ImportError:
                print("  [WARNING] tomli not installed, skipping strict parse check.")
                return

        with open("pyproject.toml", "rb") as f:
            data = toml.load(f)
            
        project = data.get("project", {})
        print(f"  Name: {project.get('name')}")
        print(f"  Version: {project.get('version')}")
        print("pyproject.toml Validation: PASSED")
        
    except Exception as e:
        print(f"  [ERROR] Parsing failed: {e}")

if __name__ == "__main__":
    # Ensure we are in project root
    if not os.path.exists("src"):
        # Try moving up if in scripts
        if os.path.basename(os.getcwd()) == "scripts":
            os.chdir("..")
            
    validate_project_structure()
    validate_pyproject()
