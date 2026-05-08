"""
SystemMonitor - Cross-platform hardware and connectivity monitor.

Two-interval design:
  check_interval    - how often to sample the system (e.g. every 10 min).
                      Every sample is written to the daily log and refreshes the
                      daily JSON with the latest raw reading.
  averaging_interval - how often to flush a summarised row to the daily CSV
                      (e.g. every 60 min).  Each CSV row is the min / max / avg
                      of all samples collected during that window only.

One file of each type is created per calendar day:
  system_YYYY-MM-DD.log            - raw text entry for every check
  latest_stats_YYYY-MM-DD.json     - latest single raw reading (overwritten each check)
  stats_YYYY-MM-DD.csv             - one row per averaging window
"""

import time
import json
import socket
import shutil
import csv
import platform
from datetime import datetime
from dataclasses import dataclass, asdict, fields
from typing import Optional, List
from pathlib import Path

try:
    import psutil
except ImportError:
    raise ImportError("The 'psutil' module is required. Install it using: pip install psutil")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RawReading:
    """A single point-in-time snapshot of system stats."""
    timestamp: str
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    storage_used_gb: float
    storage_total_gb: float
    storage_percent: float
    cpu_temp_c: Optional[float]
    internet_connected: bool


@dataclass
class AveragedRow:
    """Aggregated stats over one averaging window, written as a single CSV row."""
    period_start: str
    period_end: str
    samples: int
    cpu_avg_percent: float
    cpu_min_percent: float
    cpu_max_percent: float
    ram_avg_percent: float
    ram_min_percent: float
    ram_max_percent: float
    ram_avg_used_gb: float
    ram_total_gb: float
    storage_avg_percent: float
    storage_used_gb: float
    storage_total_gb: float
    cpu_temp_avg_c: Optional[float]
    cpu_temp_min_c: Optional[float]
    cpu_temp_max_c: Optional[float]
    internet_uptime_pct: float   # % of samples where internet was UP


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class SystemMonitor:
    """
    Samples system vitals every `check_interval_seconds`.

    - Each sample  →  appended to the daily .log file
                   →  overwrites the daily .json file
    - Every `averaging_interval_seconds` worth of samples  →  one row in the
      daily .csv file, containing min/max/avg of that window only.

    All output files are named with the current date (YYYY-MM-DD) and stored
    in `output_dir`.  A new set of files is created automatically at midnight.
    """

    CSV_COLUMNS = [f.name for f in fields(AveragedRow)]

    def __init__(
        self,
        check_interval_seconds: int = 600,       # 10 min default
        averaging_interval_seconds: int = 3600,  # 60 min default
        output_dir: str = ".",
        storage_path: str = "/"
    ):
        if averaging_interval_seconds < check_interval_seconds:
            raise ValueError(
                "averaging_interval_seconds must be >= check_interval_seconds"
            )

        self.check_interval = check_interval_seconds
        self.averaging_interval = averaging_interval_seconds
        self.output_dir = Path(output_dir).resolve()
        self.storage_path = str(Path(storage_path).resolve())

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._platform = platform.system()

        # Buffer that accumulates raw readings until the averaging window closes
        self._buffer: List[RawReading] = []
        self._window_start: Optional[datetime] = None

    # ------------------------------------------------------------------
    # File-path helpers  (date-stamped, refreshed per call)
    # ------------------------------------------------------------------

    def _dated(self, prefix: str, ext: str) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        return self.output_dir / f"{prefix}_{today}.{ext}"

    @property
    def _log_path(self) -> Path:
        return self._dated("system", "log")

    @property
    def _json_path(self) -> Path:
        return self._dated("latest_stats", "json")

    @property
    def _csv_path(self) -> Path:
        return self._dated("stats", "csv")

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------

    def _check_internet(self) -> bool:
        """TCP probe against Google's public DNS (8.8.8.8:53)."""
        try:
            socket.setdefaulttimeout(3)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("8.8.8.8", 53))
            return True
        except OSError:
            return False

    def _get_cpu_temperature(self) -> Optional[float]:
        """Returns the CPU temperature in °C, or None if unavailable."""
        if not hasattr(psutil, "sensors_temperatures"):
            return None
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            for name in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                if name in temps and temps[name]:
                    return temps[name][0].current
            first = list(temps.values())[0]
            if first:
                return first[0].current
        except Exception:
            pass
        return None

    def _sample(self) -> RawReading:
        """Collect one raw reading of all system stats."""
        cpu = psutil.cpu_percent(interval=None)

        mem = psutil.virtual_memory()
        ram_used  = mem.used  / (1024 ** 3)
        ram_total = mem.total / (1024 ** 3)

        disk = shutil.disk_usage(self.storage_path)
        stor_used  = disk.used  / (1024 ** 3)
        stor_total = disk.total / (1024 ** 3)
        stor_pct   = (disk.used / disk.total * 100) if disk.total else 0.0

        temp     = self._get_cpu_temperature()
        internet = self._check_internet()

        return RawReading(
            timestamp=datetime.now().isoformat(),
            cpu_percent=round(cpu, 1),
            ram_used_gb=round(ram_used, 2),
            ram_total_gb=round(ram_total, 2),
            ram_percent=round(mem.percent, 1),
            storage_used_gb=round(stor_used, 2),
            storage_total_gb=round(stor_total, 2),
            storage_percent=round(stor_pct, 1),
            cpu_temp_c=round(temp, 1) if temp is not None else None,
            internet_connected=internet,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, readings: List[RawReading], window_end: datetime) -> AveragedRow:
        """Reduce a list of raw readings into one AveragedRow."""
        n = len(readings)

        cpu_vals  = [r.cpu_percent for r in readings]
        ram_vals  = [r.ram_percent for r in readings]
        ram_used  = [r.ram_used_gb for r in readings]
        temp_vals = [r.cpu_temp_c  for r in readings if r.cpu_temp_c is not None]
        inet_ups  = sum(1 for r in readings if r.internet_connected)

        def avg(lst): return round(sum(lst) / len(lst), 1) if lst else None

        return AveragedRow(
            period_start=readings[0].timestamp,
            period_end=window_end.isoformat(),
            samples=n,
            cpu_avg_percent=avg(cpu_vals),
            cpu_min_percent=round(min(cpu_vals), 1),
            cpu_max_percent=round(max(cpu_vals), 1),
            ram_avg_percent=avg(ram_vals),
            ram_min_percent=round(min(ram_vals), 1),
            ram_max_percent=round(max(ram_vals), 1),
            ram_avg_used_gb=round(sum(ram_used) / n, 2),
            ram_total_gb=readings[-1].ram_total_gb,
            storage_avg_percent=round(sum(r.storage_percent for r in readings) / n, 1),
            storage_used_gb=readings[-1].storage_used_gb,
            storage_total_gb=readings[-1].storage_total_gb,
            cpu_temp_avg_c=avg(temp_vals),
            cpu_temp_min_c=round(min(temp_vals), 1) if temp_vals else None,
            cpu_temp_max_c=round(max(temp_vals), 1) if temp_vals else None,
            internet_uptime_pct=round(inet_ups / n * 100, 1),
        )

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_log(self, r: RawReading) -> None:
        """Append one raw reading to the daily log file."""
        line = (
            f"[{r.timestamp}] "
            f"CPU: {r.cpu_percent}% | "
            f"RAM: {r.ram_percent}% ({r.ram_used_gb}GB/{r.ram_total_gb}GB) | "
            f"Storage: {r.storage_percent}% ({r.storage_used_gb}GB/{r.storage_total_gb}GB) | "
            f"Temp: {r.cpu_temp_c}°C | "
            f"Net: {'UP' if r.internet_connected else 'DOWN'}\n"
        )
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _write_json(self, r: RawReading) -> None:
        """Overwrite the daily JSON file with the latest raw reading."""
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(r), f, indent=4)

    def _write_csv_row(self, row: AveragedRow) -> None:
        """Append one averaged row to the daily CSV, creating a header if new."""
        path = self._csv_path
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(row))

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the infinite monitoring loop."""
        checks_per_window = self.averaging_interval // self.check_interval

        print("=" * 60)
        print("SystemMonitor started")
        print(f"  Output directory    : {self.output_dir}")
        print(f"  Check interval      : {self.check_interval}s")
        print(f"  Averaging interval  : {self.averaging_interval}s  "
              f"({checks_per_window} samples per CSV row)")
        print(f"  Storage path        : {self.storage_path}")
        print("=" * 60)
        print("Log  → system_YYYY-MM-DD.log         (every check)")
        print("JSON → latest_stats_YYYY-MM-DD.json  (every check)")
        print("CSV  → stats_YYYY-MM-DD.csv           (every averaging window)")
        print("Press Ctrl+C to stop.\n")

        # Prime psutil so the very first cpu_percent call is meaningful
        psutil.cpu_percent(interval=None)
        time.sleep(1)

        self._window_start = datetime.now()
        next_check_time = time.monotonic()

        try:
            while True:
                # --- sample ---
                reading = self._sample()
                self._buffer.append(reading)

                # --- per-check outputs ---
                self._write_log(reading)
                self._write_json(reading)

                print(
                    f"[{reading.timestamp}] CHECK  "
                    f"CPU: {reading.cpu_percent}%  "
                    f"RAM: {reading.ram_percent}%  "
                    f"Temp: {reading.cpu_temp_c}°C  "
                    f"Net: {'UP' if reading.internet_connected else 'DOWN'}  "
                    f"(buffer: {len(self._buffer)}/{checks_per_window})"
                )

                # --- averaging window flush ---
                elapsed = (datetime.now() - self._window_start).total_seconds()
                if elapsed >= self.averaging_interval:
                    window_end = datetime.now()
                    row = self._aggregate(self._buffer, window_end)
                    self._write_csv_row(row)

                    print(
                        f"[{window_end.isoformat()}] CSV ROW  "
                        f"samples={row.samples}  "
                        f"cpu_avg={row.cpu_avg_percent}%  "
                        f"cpu_min={row.cpu_min_percent}%  "
                        f"cpu_max={row.cpu_max_percent}%  "
                        f"ram_avg={row.ram_avg_percent}%  "
                        f"inet_up={row.internet_uptime_pct}%"
                    )

                    # Reset for next window
                    self._buffer = []
                    self._window_start = window_end

                # --- sleep until the next check is due ---
                next_check_time += self.check_interval
                sleep_for = next_check_time - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            if self._buffer:
                print(f"  {len(self._buffer)} buffered sample(s) were not flushed to CSV "
                      "(averaging window was not complete).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    monitor = SystemMonitor(
        check_interval_seconds=10,       # sample every 10 minutes
        averaging_interval_seconds=60,  # write one CSV row every 60 minutes
        output_dir="output",              # all files land here
        storage_path="/",                 # partition to monitor
    )
    monitor.run()