import os
import sys
import time
import shutil
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from krazy_logger import KrazyLogger

logger = KrazyLogger("storage_monitor").get_logger(
    log_file="logs/storage_monitor.log",
    rotation_type="time",
)


class ThresholdType(str, Enum):
    GB          = "gb"
    PERCENTAGE  = "percentage"


@dataclass
class StorageConfig:
    path: str                               = "recordings"   # directory to monitor
    threshold_type: ThresholdType           = ThresholdType.GB
    threshold_value: float                  = 2.0            # GB  or  % free required
    poll_interval_sec: float                = 30.0           # how often to re-check
    recovery_headroom_multiplier: float     = 1.5            # resume only when free >=
                                                             # threshold * this multiplier


class StorageState(str, Enum):
    OK       = "ok"
    LOW      = "low"
    UNKNOWN  = "unknown"


class StorageMonitor:
    """
    Continuously monitors free disk space for a given path.

    Consumers call `is_ok()` to check whether recording is allowed.
    Internally the monitor polls the filesystem every `poll_interval_sec`
    seconds and transitions between StorageState.OK and StorageState.LOW.

    Resume threshold = threshold_value * recovery_headroom_multiplier
    so that the system does not oscillate between start/stop rapidly.
    """

    def __init__(self, config: Optional[StorageConfig] = None) -> None:
        self.config: StorageConfig = config or StorageConfig()
        self.state: StorageState = StorageState.UNKNOWN

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        os.makedirs(self.config.path, exist_ok=True)
        self._stop_event.clear()
        # Do one synchronous check before spawning the thread so callers
        # get a valid state immediately after start().
        self._check()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="storage-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[StorageMonitor] Started — path=%r  threshold=%s %s  poll=%.0fs",
            self.config.path,
            self.config.threshold_value,
            self.config.threshold_type.value,
            self.config.poll_interval_sec,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[StorageMonitor] Stopped.")

    def is_ok(self) -> bool:
        """Returns True when free space is above the threshold."""
        return self.state == StorageState.OK

    def get_free_gb(self) -> float:
        usage = shutil.disk_usage(self.config.path)
        return usage.free / (1024 ** 3)

    def get_free_percent(self) -> float:
        usage = shutil.disk_usage(self.config.path)
        return (usage.free / usage.total) * 100.0

    def get_stats(self) -> dict:
        usage = shutil.disk_usage(self.config.path)
        return {
            "total_gb":   round(usage.total / (1024 ** 3), 2),
            "used_gb":    round(usage.used  / (1024 ** 3), 2),
            "free_gb":    round(usage.free  / (1024 ** 3), 2),
            "free_pct":   round((usage.free / usage.total) * 100.0, 2),
            "state":      self.state.value,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _free_value(self) -> float:
        """Returns the current free value in the configured unit."""
        if self.config.threshold_type == ThresholdType.GB:
            return self.get_free_gb()
        return self.get_free_percent()

    def _check(self) -> None:
        try:
            free = self._free_value()
            threshold = self.config.threshold_value
            resume_threshold = threshold * self.config.recovery_headroom_multiplier
            unit = self.config.threshold_type.value

            previous_state = self.state

            if self.state == StorageState.LOW:
                # Only recover once free space exceeds the resume threshold
                if free >= resume_threshold:
                    self.state = StorageState.OK
                    logger.info(
                        "[StorageMonitor] Storage recovered — free=%.2f %s (resume threshold=%.2f %s).",
                        free, unit, resume_threshold, unit,
                    )
                    logger.notify_success(
                        "Storage Recovered",
                        f"Free space: {free:.2f} {unit}. Tasks will resume.",
                    )
            else:
                if free < threshold:
                    self.state = StorageState.LOW
                    logger.warning(
                        "[StorageMonitor] Storage LOW — free=%.2f %s (threshold=%.2f %s).",
                        free, unit, threshold, unit,
                    )
                    logger.notify_error(
                        "Storage Low",
                        f"Free space: {free:.2f} {unit}. Tasks paused.",
                    )
                else:
                    self.state = StorageState.OK

            if previous_state == StorageState.UNKNOWN:
                stats = self.get_stats()
                logger.info(
                    "[StorageMonitor] Initial check — free=%.2f GB (%.1f%%)  state=%s",
                    stats["free_gb"], stats["free_pct"], self.state.value,
                )

        except Exception as exc:
            logger.error("[StorageMonitor] Failed to check disk usage: %s", exc)
            self.state = StorageState.UNKNOWN

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.config.poll_interval_sec)
            if not self._stop_event.is_set():
                self._check()
        


if __name__ == "__main__":

    storage_cfg = StorageConfig(
        path="./",
        threshold_type=ThresholdType.GB,
        threshold_value=50,
        recovery_headroom_multiplier=1.5,
        poll_interval_sec=5,
    )

    storage_monitor = StorageMonitor(config=storage_cfg)
    storage_monitor.start()

    import signal

    def _handle_exit(sig, frame):
        print(f"Received exit signal {sig}. Stopping monitor...")
        storage_monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    while not storage_monitor._stop_event.is_set():
        storage_monitor._stop_event.wait(timeout=5)
        stats = storage_monitor.get_stats()
        print(f"Free space: {stats['free_gb']} GB ({stats['free_pct']}%)  State: {stats['state']}")