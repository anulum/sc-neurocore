# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker process tree under real kernel refusals

"""Kernel refusals leave custody honest: nothing is claimed that did not happen.

Each case runs the real tree code in a child interpreter whose seccomp filter
makes the kernel refuse one system call, as a hardened service manager or a
security module can. Races between observation and a descendant's exit are
not scheduled here; they belong to the adversarial end-to-end proof.
"""

from __future__ import annotations

import errno

import pytest

from tests.studio_seccomp_support import SECCOMP_AVAILABLE, Refusal, run_refused

pytestmark = pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="seccomp filters need Linux x86_64")

_PR_SET_CHILD_SUBREAPER = 36


def test_refused_subreaper_reports_the_kernel_errno() -> None:
    """A refused ``prctl`` raises the kernel errno instead of claiming custody."""
    result = run_refused(
        "from sc_neurocore.studio.platform.storage_worker_tree import become_child_subreaper\n"
        "install_refusals(REFUSALS)\n"
        "try:\n"
        "    become_child_subreaper()\n"
        "    outcome = None\n"
        "except OSError as refused:\n"
        "    outcome = refused.errno\n"
        "print(json.dumps({'errno': outcome}))\n",
        [Refusal("prctl", errno.EPERM, _PR_SET_CHILD_SUBREAPER)],
    )
    assert result == {"errno": errno.EPERM}


def test_refused_signal_delivery_is_an_unconfirmed_stop() -> None:
    """Members the kernel will not signal stay live and the stop is refused."""
    result = run_refused(
        "import os, signal\n"
        "from tests.studio_storage_tree_support import observe_until, one_child_leader\n"
        "child, tree, grand = one_child_leader()\n"
        "observe_until(tree, [grand])\n"
        "install_refusals(REFUSALS)\n"
        "stopped = tree.kill(rounds=2)\n"
        "live = sorted(member.pid for member in tree.live())\n"
        "os.killpg(child.pid, signal.SIGKILL)\n"
        "child.wait(timeout=10.0)\n"
        "tree.close()\n"
        "print(json.dumps({'stopped': stopped, 'live': live == sorted([child.pid, grand])}))\n",
        [Refusal("pidfd_send_signal", errno.EPERM)],
    )
    assert result == {"stopped": False, "live": True}


def test_refused_kill_leaves_an_adopted_process_unreported() -> None:
    """An adopted process the kernel will not kill is not reported as killed."""
    result = run_refused(
        "import os, signal, time\n"
        "from sc_neurocore.studio.platform.storage_worker_tree import (\n"
        "    become_child_subreaper, reap_adopted)\n"
        "become_child_subreaper()\n"
        "read, write = os.pipe()\n"
        "helper = os.fork()\n"
        "if helper == 0:\n"
        "    os.setsid()\n"
        "    stray = os.fork()\n"
        "    if stray == 0:\n"
        "        time.sleep(60)\n"
        "        os._exit(0)\n"
        "    os.write(write, str(stray).encode())\n"
        "    os._exit(0)\n"
        "os.waitpid(helper, 0)\n"
        "stray = int(os.read(read, 32))\n"
        "install_refusals(REFUSALS)\n"
        "killed = reap_adopted([], leaders=set())\n"
        "with open(f'/proc/{stray}/stat') as handle:\n"
        "    state = handle.read().rsplit(')', 1)[-1].split()[0]\n"
        "descriptor = os.pidfd_open(stray)\n"
        "signal.pidfd_send_signal(descriptor, signal.SIGKILL)\n"
        "os.waitpid(stray, 0)\n"
        "os.close(descriptor)\n"
        "print(json.dumps({'killed': list(killed), 'alive': state != 'Z'}))\n",
        [Refusal("kill", errno.EPERM)],
    )
    assert result == {"killed": [], "alive": True}
