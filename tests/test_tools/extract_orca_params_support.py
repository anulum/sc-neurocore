# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former extract_orca_params

from __future__ import annotations

"""Unit and CLI tests for ``tools/quantum/extract_orca_params.py``.

The fixtures are compact synthetic ORCA fragments that reproduce the exact
line layout of an ORCA 6.1 ``EPRNMR`` output (gtensor + hyperfine) for one
phosphorus and one calcium nucleus, so the parser is exercised without
shipping a multi-hundred-kilobyte real output blob."""

import hashlib


import importlib.util


import json


import subprocess


import sys


from pathlib import Path


from types import ModuleType


import pytest


REPO = Path(__file__).resolve().parents[2]


TOOL = REPO / "tools/quantum/extract_orca_params.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extract_orca_params", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_G_MATRIX_BLOCK = """\
ELECTRONIC G-MATRIX
-------------------

The g-matrix:
              2.0251815   -0.0060895    0.0127598
             -0.0063840    2.0679999   -0.0347135
              0.0131680   -0.0328507    2.0382556
 Breakdown of the contributions
 gel          2.0023193    2.0023193    2.0023193
             ----------   ----------   ----------
 g(tot)       2.0114962    2.0276218    2.0923195 iso=  2.0438125
 Delta-g      0.0091769    0.0253025    0.0900002 iso=  0.0414932
"""


_P_BLOCK = """\
 Nucleus   9P : A  : Isotope=   31 I=  0.5 P=216.1834 MHz/au**3
                HFC: iso  =YES dip=YES orb=YES gauge=YES

 Total HFC matrix (all values in MHz):
               -33.5215              -0.5928              -0.8634
                -0.4863             -30.3245               0.9557
                -0.8853               1.0367             -31.3777

 A(FC)         -31.8049             -31.8049             -31.8049
 A(SD)           2.3221              -0.1421              -2.1800
 A(ORB+DIA)     -0.0292               0.0768               0.1435    A(PC) =    0.0637
 A(ORB)         -0.0297               0.0765               0.1431    A(PC) =    0.0633
 A(DIA)          0.0005               0.0003               0.0004    A(PC) =    0.0004
             ----------           ----------           ----------
 A(Tot)        -29.5120             -31.8702             -33.8414    A(iso)=  -31.7412
"""


_CA_BLOCK = """\
 Nucleus   0Ca: A  : Isotope=   43 I=  3.5 P=-35.9513 MHz/au**3
                HFC: iso  =YES dip=YES orb= NO gauge= NO

 Total HFC matrix (all values in MHz):
                 1.0548              -0.1123              -0.1961
                -0.1123               1.3845              -0.0915
                -0.1961              -0.0915               1.2675

 A(FC)           1.2356               1.2356               1.2356
 A(SD)          -0.3397               0.1405               0.1992
             ----------           ----------           ----------
 A(Tot)          0.8959               1.3761               1.4348    A(iso)=    1.2356
"""


def _full_output(
    *,
    g_block: str = _G_MATRIX_BLOCK,
    p_block: str = _P_BLOCK,
    ca_block: str = _CA_BLOCK,
    final_energy_line: str = "FINAL SINGLE POINT ENERGY     -9953.726192774189\n",
    run_time_line: str = ("TOTAL RUN TIME: 0 days 4 hours 0 minutes 52 seconds 740 msec\n"),
    terminated: bool = True,
) -> str:
    parts = [
        "                         Program Version 6.1.1  -  RELEASE   -\n",
        "|  1> ! UKS B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 SP\n",
        "|  7> * xyzfile 1 2 input.xyz\n",
        "General Settings:\n",
        " Hartree-Fock type      HFTyp           .... UHF\n",
        " Total Charge           Charge          ....    1\n",
        " Multiplicity           Mult            ....    2\n",
        " Number of Electrons    NEL             ....  461\n",
        " Basis Dimension        Dim             .... 1290\n",
        final_energy_line,
        g_block,
        "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE (15 nuclei)\n",
        "Energy             : -9953.4792582384379784 Eh\n",
        p_block,
        ca_block,
    ]
    if terminated:
        parts.append("                             ****ORCA TERMINATED NORMALLY****\n")
        parts.append(run_time_line)
    return "".join(parts)


def _write_output(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


__all__ = [
    "hashlib",
    "importlib",
    "json",
    "subprocess",
    "sys",
    "Path",
    "ModuleType",
    "pytest",
    "REPO",
    "TOOL",
    "_load_tool",
    "_G_MATRIX_BLOCK",
    "_P_BLOCK",
    "_CA_BLOCK",
    "_full_output",
    "_write_output",
]
