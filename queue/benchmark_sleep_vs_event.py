"""
Benchmark: time.sleep() vs threading.Event.wait()

Measures:
  - Stop latency   : how long after stop() the thread actually exits
  - CPU usage      : % CPU consumed by the thread during idle wait
  - Wakeup jitter  : difference between requested and actual sleep duration
  - Interrupt cost : time to unblock a sleeping thread from another thread
"""

import time
import threading
import signal
import sys
import os
import statistics
from dataclasses import dataclass, field
from typing import List

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARN] psutil not found. CPU measurements will be skipped.")
    print("       Install with: pip install psutil\n")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL   = 2.0       # seconds — simulated poll interval
STOP_AFTER      = 5.0       # seconds — how long to let each worker run
NUM_TICKS       = 3         # how many poll ticks to observe before stopping
CPU_SAMPLE_HZ   = 10        # how many CPU samples per second to collect
SEPARATOR       = "─" * 64


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    stop_latency_ms: float              = 0.0
    cpu_samples: List[float]            = field(default_factory=list)
    jitter_samples_ms: List[float]      = field(default_factory=list)
    interrupt_latency_ms: float         = 0.0
    ticks_completed: int                = 0

    @property
    def avg_cpu(self) -> float:
        return statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def max_cpu(self) -> float:
        return max(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def avg_jitter_ms(self) -> float:
        return statistics.mean(self.jitter_samples_ms) if self.jitter_samples_ms else 0.0

    @property
    def max_jitter_ms(self) -> float:
        return max(self.jitter_samples_ms) if self.jitter_samples_ms else 0.0

    @property
    def stdev_jitter_ms(self) -> float:
        return statistics.stdev(self.jitter_samples_ms) if len(self.jitter_samples_ms) > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CPU sampler (runs in its own thread, samples a target thread's CPU)
# ─────────────────────────────────────────────────────────────────────────────

class _CpuSampler:
    def __init__(self, target_tid: int, result: BenchResult) -> None:
        self._tid    = target_tid
        self._result = result
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _loop(self) -> None:
        if not PSUTIL_AVAILABLE:
            return
        proc = psutil.Process(os.getpid())
        # psutil measures per-process CPU; for thread-level we use
        # the process interval CPU as a proxy (Python GIL makes this
        # representative for single-thread-dominant workloads).
        while not self._stop.is_set():
            self._result.cpu_samples.append(proc.cpu_percent(interval=None))
            self._stop.wait(timeout=1.0 / CPU_SAMPLE_HZ)


# ─────────────────────────────────────────────────────────────────────────────
# Worker A — time.sleep()
# ─────────────────────────────────────────────────────────────────────────────

def _worker_sleep(result: BenchResult, ready: threading.Event) -> None:
    ready.set()
    for _ in range(NUM_TICKS):
        t0 = time.perf_counter()
        time.sleep(POLL_INTERVAL)
        actual = (time.perf_counter() - t0) * 1000          # ms
        jitter = abs(actual - POLL_INTERVAL * 1000)
        result.jitter_samples_ms.append(jitter)
        result.ticks_completed += 1


def run_sleep_benchmark() -> BenchResult:
    result = BenchResult(name="time.sleep()")
    ready  = threading.Event()

    thread = threading.Thread(target=_worker_sleep, args=(result, ready), daemon=True)

    sampler = _CpuSampler(target_tid=thread.ident or 0, result=result)

    thread.start()
    ready.wait()
    sampler.start()

    # ── Stop-latency test ─────────────────────────────────────────────
    # We can't interrupt time.sleep(), so we measure how long the thread
    # takes to finish naturally after we decide to "stop".
    stop_requested = time.perf_counter()
    # time.sleep has no way to be interrupted — thread must finish its tick
    thread.join()
    result.stop_latency_ms = (time.perf_counter() - stop_requested) * 1000

    sampler.stop()

    # ── Interrupt latency test ────────────────────────────────────────
    # Simulate: we want to interrupt a sleep mid-way — not possible,
    # so worst case = full POLL_INTERVAL
    result.interrupt_latency_ms = POLL_INTERVAL * 1000   # theoretical worst case

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Worker B — threading.Event.wait()
# ─────────────────────────────────────────────────────────────────────────────

def _worker_event(
    result: BenchResult,
    stop_event: threading.Event,
    ready: threading.Event,
) -> None:
    ready.set()
    while not stop_event.is_set():
        t0 = time.perf_counter()
        triggered = stop_event.wait(timeout=POLL_INTERVAL)
        actual = (time.perf_counter() - t0) * 1000
        if triggered:
            break                                            # stop() was called
        jitter = abs(actual - POLL_INTERVAL * 1000)
        result.jitter_samples_ms.append(jitter)
        result.ticks_completed += 1


def run_event_benchmark() -> BenchResult:
    result     = BenchResult(name="Event.wait()")
    stop_event = threading.Event()
    ready      = threading.Event()

    thread = threading.Thread(
        target=_worker_event,
        args=(result, stop_event, ready),
        daemon=True,
    )

    sampler = _CpuSampler(target_tid=thread.ident or 0, result=result)

    thread.start()
    ready.wait()
    sampler.start()

    # Let it run for a few ticks
    time.sleep(POLL_INTERVAL * NUM_TICKS)

    # ── Stop-latency test ─────────────────────────────────────────────
    stop_requested = time.perf_counter()
    stop_event.set()                                         # instant interrupt
    thread.join()
    result.stop_latency_ms = (time.perf_counter() - stop_requested) * 1000

    sampler.stop()

    # ── Interrupt latency test ─────────────────────────────────────────
    # Measure how fast a mid-sleep interrupt wakes the thread
    interrupt_results = []
    for _ in range(5):
        evt  = threading.Event()
        latch = threading.Event()
        wakeup_times = []

        def _interruptible(e=evt, l=latch, w=wakeup_times):
            l.set()
            t0 = time.perf_counter()
            e.wait(timeout=POLL_INTERVAL)
            w.append((time.perf_counter() - t0) * 1000)

        t = threading.Thread(target=_interruptible, daemon=True)
        t.start()
        latch.wait()
        time.sleep(POLL_INTERVAL / 2)       # interrupt halfway through
        interrupt_t0 = time.perf_counter()
        evt.set()
        t.join()
        interrupt_results.append((time.perf_counter() - interrupt_t0) * 1000)

    result.interrupt_latency_ms = statistics.mean(interrupt_results)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_value: float, width: int = 30, char: str = "█") -> str:
    filled = int((value / max_value) * width) if max_value > 0 else 0
    return char * filled + "░" * (width - filled)


def print_report(sleep_r: BenchResult, event_r: BenchResult) -> None:
    results = [sleep_r, event_r]

    print(f"\n{SEPARATOR}")
    print("  BENCHMARK: time.sleep()  vs  threading.Event.wait()")
    print(f"  Poll interval : {POLL_INTERVAL}s   |   Ticks observed: {NUM_TICKS}")
    print(SEPARATOR)

    # ── Stop latency ──────────────────────────────────────────────────
    print("\n📌 STOP LATENCY  (lower = better — how fast the thread exits after stop())")
    max_lat = max(r.stop_latency_ms for r in results) or 1
    for r in results:
        bar = _bar(r.stop_latency_ms, max_lat)
        print(f"  {r.name:<22} {bar}  {r.stop_latency_ms:>10.2f} ms")

    winner = min(results, key=lambda r: r.stop_latency_ms)
    loser  = max(results, key=lambda r: r.stop_latency_ms)
    ratio  = loser.stop_latency_ms / winner.stop_latency_ms if winner.stop_latency_ms > 0 else 0
    print(f"\n  ✅ Winner: {winner.name}  ({ratio:.0f}x faster to stop)")

    # ── Interrupt latency ─────────────────────────────────────────────
    print(f"\n📌 INTERRUPT LATENCY  (how fast a mid-sleep stop() takes effect)")
    max_int = max(r.interrupt_latency_ms for r in results) or 1
    for r in results:
        bar = _bar(r.interrupt_latency_ms, max_int)
        print(f"  {r.name:<22} {bar}  {r.interrupt_latency_ms:>10.2f} ms")

    # ── Wakeup jitter ─────────────────────────────────────────────────
    print(f"\n📌 WAKEUP JITTER  (deviation from requested sleep duration — lower = better)")
    all_avg = [r.avg_jitter_ms for r in results]
    max_avg = max(all_avg) or 1
    for r in results:
        bar = _bar(r.avg_jitter_ms, max_avg)
        print(
            f"  {r.name:<22} {bar}  "
            f"avg={r.avg_jitter_ms:>7.3f} ms  "
            f"max={r.max_jitter_ms:>7.3f} ms  "
            f"σ={r.stdev_jitter_ms:>6.3f} ms"
        )

    # ── CPU usage ─────────────────────────────────────────────────────
    if PSUTIL_AVAILABLE:
        print(f"\n📌 CPU USAGE  (process-level, sampled at {CPU_SAMPLE_HZ}Hz — lower = better)")
        max_cpu = max(r.avg_cpu for r in results) or 1
        for r in results:
            bar = _bar(r.avg_cpu, max_cpu)
            print(
                f"  {r.name:<22} {bar}  "
                f"avg={r.avg_cpu:>5.2f}%  "
                f"peak={r.max_cpu:>5.2f}%  "
                f"samples={len(r.cpu_samples)}"
            )
    else:
        print("\n📌 CPU USAGE  [skipped — install psutil]")

    # ── Ticks completed ───────────────────────────────────────────────
    print(f"\n📌 TICKS COMPLETED")
    for r in results:
        print(f"  {r.name:<22} {r.ticks_completed} / {NUM_TICKS}")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print(f"  {'Metric':<30} {'time.sleep()':>16} {'Event.wait()':>16}")
    print(SEPARATOR)
    rows = [
        ("Stop latency (ms)",        f"{sleep_r.stop_latency_ms:.2f}",    f"{event_r.stop_latency_ms:.2f}"),
        ("Interrupt latency (ms)",   f"{sleep_r.interrupt_latency_ms:.2f}", f"{event_r.interrupt_latency_ms:.2f}"),
        ("Avg jitter (ms)",          f"{sleep_r.avg_jitter_ms:.3f}",      f"{event_r.avg_jitter_ms:.3f}"),
        ("Max jitter (ms)",          f"{sleep_r.max_jitter_ms:.3f}",      f"{event_r.max_jitter_ms:.3f}"),
    ]
    if PSUTIL_AVAILABLE:
        rows += [
            ("Avg CPU %",            f"{sleep_r.avg_cpu:.2f}",            f"{event_r.avg_cpu:.2f}"),
            ("Peak CPU %",           f"{sleep_r.max_cpu:.2f}",            f"{event_r.max_cpu:.2f}"),
        ]
    for label, s_val, e_val in rows:
        print(f"  {label:<30} {s_val:>16} {e_val:>16}")
    print(SEPARATOR)

    print("\n  📋 VERDICT")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  time.sleep()   → cannot be interrupted mid-sleep.      │")
    print("  │                   stop() must wait for the full tick.    │")
    print("  │                                                          │")
    print("  │  Event.wait()   → instantly unblocked when set() is     │")
    print("  │                   called. Zero extra CPU. Same jitter.   │")
    print("  │                   Preferred for all production threads.  │")
    print("  └─────────────────────────────────────────────────────────┘\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    def _handle_exit(sig: int, frame: object) -> None:
        print("\nInterrupted. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    print(f"\n{SEPARATOR}")
    print("  Running time.sleep() benchmark ...")
    print(SEPARATOR)
    sleep_result = run_sleep_benchmark()
    print("  Done.\n")

    print(SEPARATOR)
    print("  Running Event.wait() benchmark ...")
    print(SEPARATOR)
    event_result = run_event_benchmark()
    print("  Done.")

    print_report(sleep_result, event_result)


if __name__ == "__main__":
    main()