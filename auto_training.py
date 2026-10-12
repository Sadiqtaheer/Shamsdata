"""Watch source CSVs and retrain dashboard models when their contents change.

Usage:
    AutoTrainer(project_dir, data_dir, report_dir, source_files).start()
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


class AutoTrainer:
    def __init__(
        self,
        project_dir: Path,
        data_dir: Path,
        reports_dir: Path,
        source_files: Mapping[str, str],
        *,
        poll_seconds: float = 5.0,
        settle_seconds: float = 15.0,
        retry_seconds: float = 60.0,
        command: list[str] | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir)
        self.source_files = dict(source_files)
        self.poll_seconds = poll_seconds
        self.settle_seconds = settle_seconds
        self.retry_seconds = retry_seconds
        self.command = command or [sys.executable, str(self.project_dir / "train.py")]
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._retry_at = 0.0
        self._status: dict[str, object] = {
            "state": "starting",
            "generation": 0,
            "error": None,
        }

    def start(self) -> AutoTrainer:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._watch,
                name="shams-auto-trainer",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_seconds * 2))

    def snapshot(self) -> dict[str, object]:
        with self._status_lock:
            return self._status.copy()

    def _set_status(self, state: str, *, error: str | None = None) -> None:
        with self._status_lock:
            self._status["state"] = state
            self._status["error"] = error

    def _source_signatures(self) -> tuple[tuple[str, int, int], ...]:
        signatures = []
        for filename in self.source_files.values():
            path = self.data_dir / filename
            stat = path.stat()
            signatures.append((filename, stat.st_size, stat.st_mtime_ns))
        return tuple(signatures)

    def _source_hashes(self) -> dict[str, str]:
        hashes = {}
        for filename in self.source_files.values():
            path = self.data_dir / filename
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[filename] = digest.hexdigest()
        return hashes

    def _manifest_hashes(self) -> dict[str, str]:
        try:
            manifest = json.loads(
                (self.reports_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return {}
        return manifest.get("source_sha256", {})

    def _wait_until_stable(
        self, signatures: tuple[tuple[str, int, int], ...]
    ) -> tuple[tuple[str, int, int], ...]:
        stable_since = time.monotonic()
        while not self._stop_event.wait(self.poll_seconds):
            try:
                current = self._source_signatures()
            except OSError:
                current = None
            if current is None or current != signatures:
                signatures = current
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= self.settle_seconds:
                return current
        return signatures

    def _set_error(self, error: Exception) -> None:
        self._retry_at = time.monotonic() + self.retry_seconds
        self._set_status("error", error=str(error))

    def _run_training(self) -> None:
        self._set_status("running")
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(
                self.command,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            log = (
                f"Started: {started_at}\nExit code: {result.returncode}\n\n"
                f"STDOUT\n{result.stdout}\nSTDERR\n{result.stderr}"
            )
            (self.reports_dir / "auto_training.log").write_text(log, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(
                    f"train.py exited with code {result.returncode}; see auto_training.log"
                )
            if self._source_hashes() != self._manifest_hashes():
                raise RuntimeError(
                    "Source files changed during training; automatic retry is scheduled."
                )
            with self._status_lock:
                self._status["state"] = "idle"
                self._status["error"] = None
                self._status["generation"] = int(self._status["generation"]) + 1
                self._status["last_run"] = datetime.now(timezone.utc).isoformat()
            self._retry_at = 0.0
        except Exception as error:
            self._set_error(error)

    def _check_and_train(self) -> None:
        try:
            if self._source_hashes() != self._manifest_hashes():
                self._run_training()
            else:
                self._set_status("idle")
                self._retry_at = 0.0
        except Exception as error:
            self._set_error(error)

    def _watch(self) -> None:
        try:
            signatures = self._source_signatures()
            if self._source_hashes() != self._manifest_hashes():
                signatures = self._wait_until_stable(signatures)
                self._check_and_train()
            else:
                self._set_status("idle")
        except Exception as error:
            signatures = None
            self._set_error(error)

        while not self._stop_event.wait(self.poll_seconds):
            try:
                current = self._source_signatures()
            except OSError as error:
                signatures = None
                self._set_error(error)
                continue

            retry_due = (
                self.snapshot()["state"] == "error"
                and time.monotonic() >= self._retry_at
            )
            if current != signatures:
                signatures = self._wait_until_stable(current)
                self._check_and_train()
            elif retry_due:
                self._check_and_train()