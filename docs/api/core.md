# Core Modules

Central infrastructure: model definition language parser, simulation orchestrator,
and tensor streaming for spike data.

## MDL Parser

::: sc_neurocore.core.mdl_parser

## Orchestrator

::: sc_neurocore.core.orchestrator

## Tensor Stream

::: sc_neurocore.core.tensor_stream

## Simulator registry

A model definition resolves its simulator by model id from the `sc_neurocore.simulators`
entry points of the installed package, so the model does not import its executor. The
`alpha` model is the first to use it. Duplicate registrations, registrations from another
distribution, load errors and a checkout without installed metadata are refused, each with a
typed error that names the cause. Refresh the editable install after a change to that entry
point group (see [install profiles](../guides/install_profiles.md)).

::: sc_neurocore.core.backends
