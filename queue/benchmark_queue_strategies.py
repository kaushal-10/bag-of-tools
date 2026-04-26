"""
Benchmark: Queue consumption strategies

Compares three patterns for consuming items from a queue:

  Strategy A — queue.get(block=True, timeout=...)
               Thread blocks inside the queue until an item arrives or timeout.

  Strategy B — queue.get_nowait() + time.sleep()
               Non-blocking get; if empty, sleep a fixed interval then retry.
               Classic "polling" pattern — simple but wasteful.

  Strategy C — queue.get_nowait() + Event.wait()
               Non-blocking get; if empty, wait on an Event that the producer
               sets when it puts an item. Zero-latency wakeup + no busy spin.

Measures:
  - CPU usage          : % CPU burned while the queue is idle
  - Consume latency    : time from item being put → item being received
  - Miss rate          : how many get() calls found an empty queue (busy spin indicator)
  - Stop latency       : how fast the consumer exits after stop signal
  - Throughput         : items consumed per second under load
"""

import queue
import time
import threading
import signal
import sys
import os
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

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

IDLE_DURATION_SEC       = 5.0       # how long to measure CPU while queue is EMPTY
PRODUCE_INTERVAL_SEC    = 0.2       # producer puts one item every N seconds
PRODUCE_COUNT           = 20        # total items the producer will put
POLL_SLEEP_SEC          = 0.01      # sleep duration for Strategy B (get_nowait + sleep)
CPU_SAMPLE_HZ           = 20        # CPU samples per second
SEPARATOR               = "─" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Strategy enum
# ─────────────────────────────────────────────────────────────────────────────

class Strategy(str, Enum):
    BLOCKING_GET    = "queue.get(blocking)"
    NOWAIT_SLEEP    = "get_nowait + sleep()"
    NOWAIT_EVENT    = "get_nowait + Event.wait()"


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    strategy: Strategy
    cpu_idle_samples: List[float]           = field(default_factory=list)
    cpu_load_samples: List[float]           = field(default_factory=list)
    consume_latencies_ms: List[float]       = field(default_factory=list)
    empty_hits: int                         = 0     # get() calls that found nothing
    total_get_calls: int                    = 0
    items_consumed: int                     = 0
    stop_latency_ms: float                  = 0.0
    throughput_items_per_sec: float         = 0.0

    # ── Derived ───────────────────────────────────────────────────────

    @property
    def avg_cpu_idle(self) -> float:
        return statistics.mean(self.cpu_idle_samples) if self.cpu_idle_samples else 0.0

    @property
    def max_cpu_idle(self) -> float:
        return max(self.cpu_idle_samples) if self.cpu_idle_samples else 0.0

    @property
    def avg_cpu_load(self) -> float:
        return statistics.mean(self.cpu_load_samples) if self.cpu_load_samples else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.consume_latencies_ms) if self.consume_latencies_ms else 0.0

    @property
    def max_latency_ms(self) -> float:
        return max(self.consume_latencies_ms) if self.consume_latencies_ms else 0.0

    @property
    def stdev_latency_ms(self) -> float:
        return (
            statistics.stdev(self.consume_latencies_ms)
            if len(self.consume_latencies_ms) > 1
            else 0.0
        )

    @property
    def miss_rate_pct(self) -> float:
        return (self.empty_hits / self.total_get_calls * 100) if self.total_get_calls > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CPU sampler
# ─────────────────────────────────────────────────────────────────────────────

class _CpuSampler:
    def __init__(self, result: BenchResult, key: str = "idle") -> None:
        """
        key: "idle"  → appends to result.cpu_idle_samples
             "load"  → appends to result.cpu_load_samples
        """
        self._result = result
        self._key    = key
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
        proc.cpu_percent(interval=None)     # discard first call (always 0)
        while not self._stop.is_set():
            sample = proc.cpu_percent(interval=None)
            if self._key == "idle":
                self._result.cpu_idle_samples.append(sample)
            else:
                self._result.cpu_load_samples.append(sample)
            self._stop.wait(timeout=1.0 / CPU_SAMPLE_HZ)


# ─────────────────────────────────────────────────────────────────────────────
# Producer  (shared across all strategies)
# ─────────────────────────────────────────────────────────────────────────────

def _run_producer(
    q: queue.Queue,
    count: int,
    interval: float,
    notify_event: Optional[threading.Event] = None,
) -> None:
    """Puts (timestamp, seq) tuples into q, then puts None as sentinel."""
    for i in range(count):
        time.sleep(interval)
        q.put((time.perf_counter(), i))
        if notify_event:
            notify_event.set()      # wake up Strategy C consumer instantly
    q.put(None)                     # sentinel


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A — queue.get(block=True, timeout=N)
# ─────────────────────────────────────────────────────────────────────────────

def _consumer_blocking(
    q: queue.Queue,
    result: BenchResult,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            result.total_get_calls += 1
            item = q.get(block=True, timeout=0.5)
            if item is None:
                break
            put_time, seq = item
            result.consume_latencies_ms.append((time.perf_counter() - put_time) * 1000)
            result.items_consumed += 1
        except queue.Empty:
            result.empty_hits += 1


def run_blocking_get() -> BenchResult:
    result      = BenchResult(strategy=Strategy.BLOCKING_GET)
    q           = queue.Queue()
    stop_event  = threading.Event()

    consumer = threading.Thread(
        target=_consumer_blocking, args=(q, result, stop_event), daemon=True
    )

    # ── Phase 1: idle CPU (empty queue) ───────────────────────────────
    consumer.start()
    sampler = _CpuSampler(result, key="idle")
    sampler.start()
    time.sleep(IDLE_DURATION_SEC)
    sampler.stop()

    # ── Phase 2: load (producer active) ───────────────────────────────
    load_sampler = _CpuSampler(result, key="load")
    load_sampler.start()
    t_start = time.perf_counter()

    producer = threading.Thread(
        target=_run_producer, args=(q, PRODUCE_COUNT, PRODUCE_INTERVAL_SEC), daemon=True
    )
    producer.start()
    producer.join()
    consumer.join(timeout=5)
    load_sampler.stop()

    result.throughput_items_per_sec = result.items_consumed / (time.perf_counter() - t_start)

    # ── Phase 3: stop latency ──────────────────────────────────────────
    q2         = queue.Queue()      # fresh empty queue
    stop_event2 = threading.Event()
    result2_ref = BenchResult(strategy=Strategy.BLOCKING_GET)
    c2 = threading.Thread(
        target=_consumer_blocking, args=(q2, result2_ref, stop_event2), daemon=True
    )
    c2.start()
    time.sleep(0.3)
    t_stop = time.perf_counter()
    stop_event2.set()
    q2.put(None)                    # unblock the get so stop_event is checked
    c2.join(timeout=3)
    result.stop_latency_ms = (time.perf_counter() - t_stop) * 1000

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B — get_nowait() + time.sleep()
# ─────────────────────────────────────────────────────────────────────────────

def _consumer_nowait_sleep(
    q: queue.Queue,
    result: BenchResult,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        result.total_get_calls += 1
        try:
            item = q.get_nowait()
            if item is None:
                break
            put_time, seq = item
            result.consume_latencies_ms.append((time.perf_counter() - put_time) * 1000)
            result.items_consumed += 1
        except queue.Empty:
            result.empty_hits += 1
            time.sleep(POLL_SLEEP_SEC)  # busy-ish polling


def run_nowait_sleep() -> BenchResult:
    result      = BenchResult(strategy=Strategy.NOWAIT_SLEEP)
    q           = queue.Queue()
    stop_event  = threading.Event()

    consumer = threading.Thread(
        target=_consumer_nowait_sleep, args=(q, result, stop_event), daemon=True
    )

    # ── Phase 1: idle CPU ──────────────────────────────────────────────
    consumer.start()
    sampler = _CpuSampler(result, key="idle")
    sampler.start()
    time.sleep(IDLE_DURATION_SEC)
    sampler.stop()

    # ── Phase 2: load ──────────────────────────────────────────────────
    load_sampler = _CpuSampler(result, key="load")
    load_sampler.start()
    t_start = time.perf_counter()

    producer = threading.Thread(
        target=_run_producer, args=(q, PRODUCE_COUNT, PRODUCE_INTERVAL_SEC), daemon=True
    )
    producer.start()
    producer.join()
    consumer.join(timeout=5)
    load_sampler.stop()

    result.throughput_items_per_sec = result.items_consumed / (time.perf_counter() - t_start)

    # ── Phase 3: stop latency ──────────────────────────────────────────
    q2          = queue.Queue()
    stop_event2 = threading.Event()
    result2_ref = BenchResult(strategy=Strategy.NOWAIT_SLEEP)
    c2 = threading.Thread(
        target=_consumer_nowait_sleep, args=(q2, result2_ref, stop_event2), daemon=True
    )
    c2.start()
    time.sleep(0.3)
    t_stop = time.perf_counter()
    stop_event2.set()
    c2.join(timeout=3)
    result.stop_latency_ms = (time.perf_counter() - t_stop) * 1000

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Strategy C — get_nowait() + Event.wait()
# ─────────────────────────────────────────────────────────────────────────────

def _consumer_nowait_event(
    q: queue.Queue,
    result: BenchResult,
    stop_event: threading.Event,
    notify_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        result.total_get_calls += 1
        try:
            item = q.get_nowait()
            if item is None:
                break
            put_time, seq = item
            result.consume_latencies_ms.append((time.perf_counter() - put_time) * 1000)
            result.items_consumed += 1
            notify_event.clear()    # reset after consuming
        except queue.Empty:
            result.empty_hits += 1
            # Park thread until producer signals or stop is requested
            notify_event.wait(timeout=1.0)
            notify_event.clear()


def run_nowait_event() -> BenchResult:
    result        = BenchResult(strategy=Strategy.NOWAIT_EVENT)
    q             = queue.Queue()
    stop_event    = threading.Event()
    notify_event  = threading.Event()

    consumer = threading.Thread(
        target=_consumer_nowait_event,
        args=(q, result, stop_event, notify_event),
        daemon=True,
    )

    # ── Phase 1: idle CPU ──────────────────────────────────────────────
    consumer.start()
    sampler = _CpuSampler(result, key="idle")
    sampler.start()
    time.sleep(IDLE_DURATION_SEC)
    sampler.stop()

    # ── Phase 2: load ──────────────────────────────────────────────────
    load_sampler = _CpuSampler(result, key="load")
    load_sampler.start()
    t_start = time.perf_counter()

    producer = threading.Thread(
        target=_run_producer,
        args=(q, PRODUCE_COUNT, PRODUCE_INTERVAL_SEC, notify_event),
        daemon=True,
    )
    producer.start()
    producer.join()
    notify_event.set()      # wake consumer to receive sentinel
    consumer.join(timeout=5)
    load_sampler.stop()

    result.throughput_items_per_sec = result.items_consumed / (time.perf_counter() - t_start)

    # ── Phase 3: stop latency ──────────────────────────────────────────
    q2            = queue.Queue()
    stop_event2   = threading.Event()
    notify_event2 = threading.Event()
    result2_ref   = BenchResult(strategy=Strategy.NOWAIT_EVENT)
    c2 = threading.Thread(
        target=_consumer_nowait_event,
        args=(q2, result2_ref, stop_event2, notify_event2),
        daemon=True,
    )
    c2.start()
    time.sleep(0.3)
    t_stop = time.perf_counter()
    stop_event2.set()
    notify_event2.set()     # unblock the wait so stop_event is checked
    c2.join(timeout=3)
    result.stop_latency_ms = (time.perf_counter() - t_stop) * 1000

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_value: float, width: int = 28, char: str = "█") -> str:
    filled = int((value / max_value) * width) if max_value > 0 else 0
    return char * filled + "░" * (width - filled)


def _winner_tag(results: list, key, lower_is_better: bool = True) -> dict:
    values = {r.strategy: key(r) for r in results}
    best   = min(values, key=values.__getitem__) if lower_is_better else max(values, key=values.__getitem__)
    return {s: (" ✅" if s == best else "") for s in values}


def print_report(results: List[BenchResult]) -> None:

    print(f"\n{SEPARATOR}")
    print("  BENCHMARK: Queue Consumption Strategies")
    print(f"  Idle phase     : {IDLE_DURATION_SEC}s  |  Producer interval : {PRODUCE_INTERVAL_SEC}s")
    print(f"  Items produced : {PRODUCE_COUNT}       |  Poll sleep (B)    : {POLL_SLEEP_SEC*1000:.0f}ms")
    print(SEPARATOR)

    # ── CPU (idle) ────────────────────────────────────────────────────
    if PSUTIL_AVAILABLE:
        print("\n📌 CPU — IDLE  (queue empty, consumer waiting — lower = better)")
        max_val = max(r.avg_cpu_idle for r in results) or 1
        tags    = _winner_tag(results, lambda r: r.avg_cpu_idle)
        for r in results:
            bar = _bar(r.avg_cpu_idle, max_val)
            print(
                f"  {r.strategy.value:<30} {bar}  "
                f"avg={r.avg_cpu_idle:>5.2f}%  peak={r.max_cpu_idle:>5.2f}%"
                f"{tags[r.strategy]}"
            )

        # ── CPU (load) ─────────────────────────────────────────────────
        print("\n📌 CPU — UNDER LOAD  (producer actively sending items)")
        max_val = max(r.avg_cpu_load for r in results) or 1
        tags    = _winner_tag(results, lambda r: r.avg_cpu_load)
        for r in results:
            bar = _bar(r.avg_cpu_load, max_val)
            print(
                f"  {r.strategy.value:<30} {bar}  "
                f"avg={r.avg_cpu_load:>5.2f}%"
                f"{tags[r.strategy]}"
            )
    else:
        print("\n📌 CPU  [skipped — install psutil]")

    # ── Consume latency ───────────────────────────────────────────────
    print("\n📌 CONSUME LATENCY  (put → received — lower = better)")
    max_val = max(r.avg_latency_ms for r in results) or 1
    tags    = _winner_tag(results, lambda r: r.avg_latency_ms)
    for r in results:
        bar = _bar(r.avg_latency_ms, max_val)
        print(
            f"  {r.strategy.value:<30} {bar}  "
            f"avg={r.avg_latency_ms:>7.3f} ms  "
            f"max={r.max_latency_ms:>7.3f} ms  "
            f"σ={r.stdev_latency_ms:>6.3f} ms"
            f"{tags[r.strategy]}"
        )

    # ── Miss rate ─────────────────────────────────────────────────────
    print("\n📌 MISS RATE  (empty get() calls / total calls — lower = better)")
    max_val = max(r.miss_rate_pct for r in results) or 1
    tags    = _winner_tag(results, lambda r: r.miss_rate_pct)
    for r in results:
        bar = _bar(r.miss_rate_pct, max_val)
        print(
            f"  {r.strategy.value:<30} {bar}  "
            f"{r.miss_rate_pct:>6.2f}%  "
            f"({r.empty_hits} misses / {r.total_get_calls} calls)"
            f"{tags[r.strategy]}"
        )

    # ── Stop latency ──────────────────────────────────────────────────
    print("\n📌 STOP LATENCY  (time from stop signal → thread exit — lower = better)")
    max_val = max(r.stop_latency_ms for r in results) or 1
    tags    = _winner_tag(results, lambda r: r.stop_latency_ms)
    for r in results:
        bar = _bar(r.stop_latency_ms, max_val)
        print(
            f"  {r.strategy.value:<30} {bar}  "
            f"{r.stop_latency_ms:>8.2f} ms"
            f"{tags[r.strategy]}"
        )

    # ── Throughput ────────────────────────────────────────────────────
    print("\n📌 THROUGHPUT  (items/sec consumed — higher = better)")
    max_val = max(r.throughput_items_per_sec for r in results) or 1
    tags    = _winner_tag(results, lambda r: r.throughput_items_per_sec, lower_is_better=False)
    for r in results:
        bar = _bar(r.throughput_items_per_sec, max_val)
        print(
            f"  {r.strategy.value:<30} {bar}  "
            f"{r.throughput_items_per_sec:>6.2f} items/s  "
            f"({r.items_consumed} consumed)"
            f"{tags[r.strategy]}"
        )

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    col = 22
    print(f"  {'Metric':<32} {'A: blocking':>{col}} {'B: nowait+sleep':>{col}} {'C: nowait+event':>{col}}")
    print(SEPARATOR)

    a, b, c = results

    rows = [
        ("Consume latency avg (ms)",  f"{a.avg_latency_ms:.3f}",    f"{b.avg_latency_ms:.3f}",    f"{c.avg_latency_ms:.3f}"),
        ("Consume latency max (ms)",  f"{a.max_latency_ms:.3f}",    f"{b.max_latency_ms:.3f}",    f"{c.max_latency_ms:.3f}"),
        ("Miss rate (%)",             f"{a.miss_rate_pct:.2f}",      f"{b.miss_rate_pct:.2f}",      f"{c.miss_rate_pct:.2f}"),
        ("Stop latency (ms)",         f"{a.stop_latency_ms:.2f}",   f"{b.stop_latency_ms:.2f}",   f"{c.stop_latency_ms:.2f}"),
        ("Throughput (items/s)",      f"{a.throughput_items_per_sec:.2f}", f"{b.throughput_items_per_sec:.2f}", f"{c.throughput_items_per_sec:.2f}"),
    ]
    if PSUTIL_AVAILABLE:
        rows += [
            ("CPU idle avg (%)",      f"{a.avg_cpu_idle:.2f}",      f"{b.avg_cpu_idle:.2f}",      f"{c.avg_cpu_idle:.2f}"),
            ("CPU idle peak (%)",     f"{a.max_cpu_idle:.2f}",      f"{b.max_cpu_idle:.2f}",      f"{c.max_cpu_idle:.2f}"),
            ("CPU load avg (%)",      f"{a.avg_cpu_load:.2f}",      f"{b.avg_cpu_load:.2f}",      f"{c.avg_cpu_load:.2f}"),
        ]

    for label, va, vb, vc in rows:
        print(f"  {label:<32} {va:>{col}} {vb:>{col}} {vc:>{col}}")
    print(SEPARATOR)

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n  📋 VERDICT")
    print("  ┌──────────────────────────────────────────────────────────────────┐")
    print("  │  A  queue.get(blocking)    → Best for simplicity. Thread parks   │")
    print("  │                              inside the queue. Zero miss rate.   │")
    print("  │                              Slight latency from timeout polling. │")
    print("  │                                                                  │")
    print("  │  B  get_nowait + sleep()   → Simplest code, worst CPU. Burns     │")
    print("  │                              cycles even when queue is empty.    │")
    print("  │                              Miss rate is very high. Avoid.      │")
    print("  │                                                                  │")
    print("  │  C  get_nowait + Event     → Best latency AND best idle CPU.     │")
    print("  │                              Producer wakes consumer instantly.  │")
    print("  │                              Ideal for high-freq producer/        │")
    print("  │                              consumer pipelines (e.g. streaming).│")
    print("  └──────────────────────────────────────────────────────────────────┘\n")


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
    print("  Strategy A: queue.get(block=True) ...")
    print(SEPARATOR)
    result_a = run_blocking_get()
    print(f"  Done. ({result_a.items_consumed} items consumed)\n")

    print(SEPARATOR)
    print("  Strategy B: get_nowait() + time.sleep() ...")
    print(SEPARATOR)
    result_b = run_nowait_sleep()
    print(f"  Done. ({result_b.items_consumed} items consumed)\n")

    print(SEPARATOR)
    print("  Strategy C: get_nowait() + Event.wait() ...")
    print(SEPARATOR)
    result_c = run_nowait_event()
    print(f"  Done. ({result_c.items_consumed} items consumed)")

    print_report([result_a, result_b, result_c])


if __name__ == "__main__":
    main()