# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network: main simulation orchestrator

"""Network: main simulation orchestrator for populations, projections, and monitors."""

from __future__ import annotations

import sys

import numpy as np

from .population import Population
from .projection import Projection
from .monitor import SpikeMonitor, StateMonitor, RateMonitor
from .stimulus import TimedArray, PoissonInput, StepCurrent


class Network:
    """Declarative network: collects objects, runs the simulation loop."""

    def __init__(self, *objects, seed=42):
        self.populations: list[Population] = []
        self.projections: list[Projection] = []
        self.spike_monitors: list[SpikeMonitor] = []
        self.state_monitors: list[StateMonitor] = []
        self.rate_monitors: list[RateMonitor] = []
        self.stimuli: list = []
        self.seed = seed
        for obj in objects:
            self.add(obj)

    def add(self, obj):
        """Register a simulation object by type."""
        if isinstance(obj, Population):
            self.populations.append(obj)
        elif isinstance(obj, Projection):
            self.projections.append(obj)
        elif isinstance(obj, SpikeMonitor):
            self.spike_monitors.append(obj)
        elif isinstance(obj, StateMonitor):
            self.state_monitors.append(obj)
        elif isinstance(obj, RateMonitor):
            self.rate_monitors.append(obj)
        elif isinstance(obj, (TimedArray, PoissonInput, StepCurrent)):
            self.stimuli.append(obj)
        else:
            raise TypeError(f"Unknown object type: {type(obj).__name__}")

    def run(self, duration, dt=0.001, progress=False):
        """Run the simulation for *duration* seconds at timestep *dt*."""
        np.random.seed(self.seed)
        n_steps = int(round(duration / dt))

        pop_to_currents = {id(p): np.zeros(p.n, dtype=np.float64) for p in self.populations}
        last_spikes = {id(p): np.zeros(p.n, dtype=np.int8) for p in self.populations}
        report_interval = max(1, n_steps // 10) if progress else 0

        for t in range(n_steps):
            if report_interval and t % report_interval == 0:
                pct = int(100 * t / n_steps)
                sys.stdout.write(f"\r[{pct:3d}%] step {t}/{n_steps}")
                sys.stdout.flush()

            for pid in pop_to_currents:
                pop_to_currents[pid][:] = 0.0

            self._apply_stimuli(pop_to_currents, t, dt)
            self._apply_projections(pop_to_currents, last_spikes)

            for pop in self.populations:
                pid = id(pop)
                spikes = pop.step_all(pop_to_currents[pid])
                last_spikes[pid] = spikes
                self._record(pop, spikes, t, dt)

            self._update_plasticity(last_spikes)

        if report_interval:
            sys.stdout.write(f"\r[100%] step {n_steps}/{n_steps}\n")
            sys.stdout.flush()

    def _apply_stimuli(self, pop_to_currents, t, dt):
        """Inject stimulus currents into target populations."""
        for stim in self.stimuli:
            target = stim.target
            if target is None:
                if self.populations:
                    target = self.populations[0]
                else:
                    continue
            pid = id(target)
            if pid not in pop_to_currents:
                continue
            if isinstance(stim, PoissonInput):
                pop_to_currents[pid][: stim.n] += stim.get_current(t)
            elif isinstance(stim, TimedArray):
                pop_to_currents[pid] += stim.get_current(t)
            elif isinstance(stim, StepCurrent):
                pop_to_currents[pid] += stim.get_current(t, dt)

    def _apply_projections(self, pop_to_currents, last_spikes):
        """Propagate spikes through all projections."""
        for proj in self.projections:
            src_spikes = last_spikes.get(id(proj.source), np.zeros(proj.source.n, dtype=np.int8))
            current = proj.propagate(src_spikes)
            pid = id(proj.target)
            if pid in pop_to_currents:
                pop_to_currents[pid] += current

    def _record(self, pop, spikes, t, dt):
        """Feed spikes/states to all monitors attached to this population."""
        for mon in self.spike_monitors:
            if mon.population is pop:
                mon.record(spikes, t)
        for mon in self.state_monitors:
            if mon.population is pop:
                mon.snapshot(t)
        for mon in self.rate_monitors:
            if mon.population is pop:
                mon.record(spikes, t, dt)

    def _update_plasticity(self, last_spikes):
        """Apply plasticity rules to projections that have them."""
        for proj in self.projections:
            if proj.plasticity:
                src_sp = last_spikes.get(id(proj.source), np.zeros(proj.source.n, dtype=np.int8))
                tgt_sp = last_spikes.get(id(proj.target), np.zeros(proj.target.n, dtype=np.int8))
                proj.update_plasticity(src_sp, tgt_sp)
