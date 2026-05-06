# Posner External Data Acquisition

These files are acquisition inputs, not runtime verification data.
Do not pass them to IBM verification until ORCA outputs have been parsed.

Current local generated ORCA output artifacts in this directory were produced
by a stopped, non-final `TightSCF` relaxation attempt. Treat those output
artifacts only as exploratory/preconditioning data. The acquisition decks in
this directory have been regenerated with `VeryTightSCF`; publication/runtime
data must come from a captured ORCA output with both `ORCA TERMINATED NORMALLY`
and `THE OPTIMIZATION HAS CONVERGED` present.

The built-in coordinates are an initial guess only; publication runs require
the completed geometry optimisation output and a documented radical state.

The neutral Ca9(PO4)6 all-electron model has 462 electrons. A neutral
doublet is invalid and is not generated. Use the cation doublet radical
workflow for electron-hole HFC data.
