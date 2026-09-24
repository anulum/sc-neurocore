# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Storage namespace descriptor verification

"""Inspect real owned directories, permission refusals and descriptor lifetimes."""

import json
import os
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_namespace import open_storage_directories


def _configuration(root: Path) -> StorageBoundaryConfiguration:
    authority = root / "authority"
    endpoint = root / "endpoint"
    authority.mkdir(mode=0o700)
    endpoint.mkdir(mode=0o755)
    return StorageBoundaryConfiguration.model_validate_json(
        json.dumps(
            {
                "storage_uid": os.getuid(),
                "api_uid": os.getuid() + 1,
                "worker_uid": os.getuid() + 2,
                "authority_root": str(authority),
                "spool_root": str(root / "spool"),
                "socket_path": str(endpoint / "storage.sock"),
                "workspace": "default",
                "frame_max_bytes": 8192,
                "max_metadata_bytes": 4096,
                "max_seed_bytes": 8192,
                "max_seed_entries": 16,
                "max_manifest_bytes": 1024,
                "max_artifact_bytes": 65536,
                "max_artifact_entries": 16,
                "transfer_timeout_seconds": 2.0,
                "max_connections": 4,
            }
        )
    )


@pytest.mark.parametrize("caller_failure", [False, True])
def test_verified_directory_handles_read_real_data_and_close(
    tmp_path: Path, caller_failure: bool
) -> None:
    """Both normal and exceptional callers release noninheritable real handles."""
    configuration = _configuration(tmp_path)
    (configuration.authority_root / "proof").write_bytes(b"retained")
    descriptors: tuple[int, int] | None = None
    try:
        with open_storage_directories(configuration) as opened:
            descriptors = (opened.authority, opened.endpoint_parent)
            assert os.fstat(opened.authority).st_ino == configuration.authority_root.stat().st_ino
            assert (
                os.fstat(opened.endpoint_parent).st_ino
                == configuration.socket_path.parent.stat().st_ino
            )
            assert all(not os.get_inheritable(fd) for fd in descriptors)
            proof = os.open("proof", os.O_RDONLY, dir_fd=opened.authority)
            try:
                assert os.read(proof, 20) == b"retained"
            finally:
                os.close(proof)
            if caller_failure:
                raise RuntimeError("caller failed")
    except RuntimeError as exc:
        assert caller_failure and str(exc) == "caller failed"
    else:
        assert not caller_failure
    assert descriptors is not None
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert (configuration.authority_root / "proof").read_bytes() == b"retained"
    assert not configuration.spool_root.exists()
    assert not configuration.socket_path.exists()


@pytest.mark.parametrize(
    "target,mode",
    [
        ("authority", 0o755),
        ("authority", 0o770),
        ("authority", 0o1700),
        ("endpoint", 0o777),
        ("endpoint", 0o1777),
        ("endpoint", 0o600),
        ("ancestor", 0o777),
    ],
)
def test_unsafe_existing_modes_refuse_without_repair(
    tmp_path: Path, target: str, mode: int
) -> None:
    """Only owned fixtures change mode; validation must not repair their permissions."""
    configuration = _configuration(tmp_path)
    path = tmp_path if target == "ancestor" else tmp_path / target
    path.chmod(mode)
    try:
        with pytest.raises(PermissionError), open_storage_directories(configuration):
            pytest.fail("unsafe namespace accepted")
        assert path.stat().st_mode & 0o7777 == mode
        assert list(configuration.authority_root.iterdir()) == []
    finally:
        path.chmod(0o700)


@pytest.mark.parametrize("fault", ["missing", "symlink", "file"])
def test_second_directory_failure_closes_all_acquired_handles(tmp_path: Path, fault: str) -> None:
    """Native open refusals do not leak the already verified authority handle."""
    configuration = _configuration(tmp_path)
    endpoint = configuration.socket_path.parent
    endpoint.rename(tmp_path / "retained")
    if fault == "symlink":
        endpoint.symlink_to(tmp_path / "retained", target_is_directory=True)
    elif fault == "file":
        endpoint.write_bytes(b"foreign")
    before = _open_descriptors()
    with pytest.raises(OSError), open_storage_directories(configuration):
        pytest.fail("replaced endpoint accepted")
    # Every descriptor opened for the attempt, including the verified
    # authority handle, is closed again.
    assert _open_descriptors() == before
    assert list(configuration.authority_root.iterdir()) == []
    assert (tmp_path / "retained").is_dir()
    if fault == "symlink":
        assert endpoint.is_symlink()
    elif fault == "file":
        assert endpoint.read_bytes() == b"foreign"


def test_identity_refusal_does_not_enter_namespace(tmp_path: Path) -> None:
    """A process that is not the configured storage identity never opens the namespace.

    This process really runs under another UID than the configured service
    identity; distinct-UID qualification itself belongs to the isolated proof.
    """
    payload = _configuration(tmp_path).model_dump(mode="json")
    payload["storage_uid"] = os.getuid() + 10
    configuration = StorageBoundaryConfiguration.model_validate_json(json.dumps(payload))
    with (
        pytest.raises(PermissionError, match="service identity"),
        open_storage_directories(configuration),
    ):
        pytest.fail("invalid service identity accepted")
    assert list(configuration.authority_root.iterdir()) == []
    assert not configuration.socket_path.exists()


@pytest.mark.parametrize("target", ["authority_root", "socket_path"])
def test_root_owned_directory_is_not_adopted_as_service_storage(
    tmp_path: Path, target: str
) -> None:
    """Read-only inspection of procfs refuses root-owned final namespaces."""
    configuration = _configuration(tmp_path)
    payload = configuration.model_dump(mode="json")
    payload[target] = "/proc" if target == "authority_root" else "/proc/storage.sock"
    configuration = StorageBoundaryConfiguration.model_validate_json(json.dumps(payload))
    with pytest.raises(PermissionError), open_storage_directories(configuration):
        pytest.fail("root-owned namespace accepted as service-owned")
    assert list((tmp_path / "authority").iterdir()) == []
    assert list((tmp_path / "endpoint").iterdir()) == []


def _open_descriptors() -> set[str]:
    """Return what every open descriptor of this process names."""
    names: set[str] = set()
    for entry in Path("/proc/self/fd").iterdir():
        try:
            names.add(f"{entry.name}:{os.readlink(entry)}")
        except FileNotFoundError:
            continue
    return names
