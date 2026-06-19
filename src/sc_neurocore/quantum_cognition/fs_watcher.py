# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GOTM filesystem watcher

"""Filesystem watcher for automatic GOTM content ingestion.

Monitors the GOTM collection directory tree for new or modified files
and auto-feeds them into a :class:`GOTMBrain` instance via the content
indexer.

Two backends are supported:
    1. **watchdog** (preferred): inotify-based, low latency.
    2. **Polling fallback**: ``os.scandir`` loop with configurable
       interval.  Used on NTFS/CIFS mounts where inotify may not
       work reliably.

Thread-safe: the watcher runs in a background daemon thread and
pushes chunks to the brain via a ``queue.Queue``.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

from .content_indexer import _EXTENSION_WEIGHTS, ContentChunk, index_file

logger = logging.getLogger(__name__)

# Try importing watchdog for native FS events
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


class GOTMWatcher:
    """Watch the GOTM collection for new/modified files.

    Parameters
    ----------
    watch_path : str or Path
        Root directory to monitor.
    repo_name : str
        Repository name for content chunks.
    debounce_s : float
        Minimum seconds between re-processing the same file.
    poll_interval_s : float
        Polling interval when using the fallback backend.
    use_polling : bool or None
        Force polling mode.  ``None`` = auto-detect (use watchdog if
        available, poll otherwise).
    """

    def __init__(
        self,
        watch_path: str | Path,
        repo_name: str = "GOTM",
        debounce_s: float = 5.0,
        poll_interval_s: float = 10.0,
        use_polling: bool | None = None,
    ) -> None:
        self.watch_path = Path(watch_path)
        if not self.watch_path.is_dir():
            raise FileNotFoundError(f"Watch path not found: {self.watch_path}")

        self.repo_name = repo_name
        self.debounce_s = debounce_s
        self.poll_interval_s = poll_interval_s

        # Decide backend
        if use_polling is None:
            self._use_polling = not HAS_WATCHDOG
        else:
            self._use_polling = use_polling

        # Queue for discovered chunks
        self._chunk_queue: queue.Queue[ContentChunk] = queue.Queue(maxsize=1000)

        # Debounce tracking: file_path → last_processed_time
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()

        # Thread state
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        logger.info(
            "GOTMWatcher: path=%s, backend=%s, debounce=%.1fs",
            self.watch_path,
            "polling" if self._use_polling else "watchdog",
            self.debounce_s,
        )

    def start(self) -> None:
        """Start the watcher in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Watcher already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="gotm-watcher", daemon=True)
        self._thread.start()
        logger.info("GOTMWatcher started")

    def stop(self) -> None:
        """Stop the watcher gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("GOTMWatcher stopped")

    def get_chunks(self, max_items: int = 50) -> list[ContentChunk]:
        """Drain the chunk queue (non-blocking).

        Parameters
        ----------
        max_items : int
            Maximum chunks to return per call.

        Returns
        -------
        list[ContentChunk]
            Newly indexed chunks since last call.
        """
        chunks: list[ContentChunk] = []
        for _ in range(max_items):
            try:
                chunks.append(self._chunk_queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    @property
    def is_running(self) -> bool:
        """Whether the watcher thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def _should_process(self, file_path: str) -> bool:
        """Check debounce timer for a file."""
        now = time.monotonic()
        with self._lock:
            last = self._last_seen.get(file_path, 0.0)
            if now - last < self.debounce_s:
                return False
            self._last_seen[file_path] = now
            return True

    def _process_file(self, file_path: Path) -> None:
        """Index a single file and enqueue its chunks."""
        if not file_path.is_file():
            return
        ext = file_path.suffix.lower()
        if ext not in _EXTENSION_WEIGHTS:
            return
        if not self._should_process(str(file_path)):
            return

        try:
            chunks = index_file(file_path, self.repo_name, self.watch_path)
            for chunk in chunks:
                try:
                    self._chunk_queue.put_nowait(chunk)
                except queue.Full:
                    logger.warning("Chunk queue full, dropping chunk from %s", file_path)
                    break
            if chunks:
                logger.debug("Indexed %d chunks from %s", len(chunks), file_path)
        except Exception as exc:
            logger.warning("Error indexing %s: %s", file_path, exc)

    def _run(self) -> None:
        """Main watcher loop (runs in background thread)."""
        if self._use_polling:
            self._run_polling()
        else:
            self._run_watchdog()

    def _run_polling(self) -> None:
        """Polling-based file watcher for NTFS compatibility."""
        # Build initial snapshot: file → mtime
        snapshot: dict[str, float] = {}
        for dirpath, dirnames, filenames in os.walk(self.watch_path):
            # Skip hidden and cache directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    snapshot[fpath] = os.path.getmtime(fpath)
                except OSError:
                    pass

        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval_s)
            if self._stop_event.is_set():
                break

            # Scan for changes
            for dirpath, dirnames, filenames in os.walk(self.watch_path):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                    except OSError:
                        continue

                    old_mtime = snapshot.get(fpath)
                    if old_mtime is None or mtime > old_mtime:
                        snapshot[fpath] = mtime
                        self._process_file(Path(fpath))

    def _run_watchdog(self) -> None:
        """Watchdog (inotify) based file watcher."""
        if not HAS_WATCHDOG:
            logger.error("watchdog not installed, falling back to polling")
            self._run_polling()
            return

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event: Any) -> None:
                if not event.is_directory:
                    watcher._process_file(Path(event.src_path))

            def on_modified(self, event: Any) -> None:
                if not event.is_directory:
                    watcher._process_file(Path(event.src_path))

        observer = Observer()
        observer.schedule(_Handler(), str(self.watch_path), recursive=True)
        observer.start()

        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(1.0)
        finally:
            observer.stop()
            observer.join(timeout=5.0)

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return (
            f"GOTMWatcher(path={self.watch_path}, "
            f"backend={'polling' if self._use_polling else 'watchdog'}, "
            f"status={status})"
        )


__all__ = ["GOTMWatcher", "HAS_WATCHDOG"]
