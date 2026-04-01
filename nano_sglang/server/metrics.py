"""
Metrics & Logging — TTFT, TBT, throughput tracking.

Collects per-request and system-wide metrics for monitoring performance.

Key metrics:
  - TTFT (Time To First Token): latency from request arrival to first output token
  - TBT (Time Between Tokens): inter-token latency during decoding
  - Throughput: tokens/second (prompt + generated)
  - Queue depth: number of waiting/running requests
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestMetrics:
    """Per-request metrics."""
    request_id: str
    prompt_tokens: int
    completion_tokens: int = 0
    arrival_time: float = 0.0
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    token_times: list[float] = field(default_factory=list)

    @property
    def ttft(self) -> Optional[float]:
        """Time To First Token (seconds)."""
        if self.first_token_time is not None:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def tbt_avg(self) -> Optional[float]:
        """Average Time Between Tokens (seconds)."""
        if len(self.token_times) < 2:
            return None
        deltas = [
            self.token_times[i] - self.token_times[i - 1]
            for i in range(1, len(self.token_times))
        ]
        return sum(deltas) / len(deltas)

    @property
    def tbt_p50(self) -> Optional[float]:
        """P50 Time Between Tokens."""
        return self._tbt_percentile(50)

    @property
    def tbt_p99(self) -> Optional[float]:
        """P99 Time Between Tokens."""
        return self._tbt_percentile(99)

    def _tbt_percentile(self, p: int) -> Optional[float]:
        if len(self.token_times) < 2:
            return None
        deltas = sorted([
            self.token_times[i] - self.token_times[i - 1]
            for i in range(1, len(self.token_times))
        ])
        idx = int(len(deltas) * p / 100)
        idx = min(idx, len(deltas) - 1)
        return deltas[idx]

    @property
    def total_time(self) -> Optional[float]:
        if self.finish_time is not None:
            return self.finish_time - self.arrival_time
        return None

    @property
    def generation_throughput(self) -> Optional[float]:
        """Tokens per second for this request's generation phase."""
        if self.total_time and self.total_time > 0 and self.completion_tokens > 0:
            return self.completion_tokens / self.total_time
        return None


class MetricsCollector:
    """
    System-wide metrics collector.

    Thread-safe. Maintains a rolling window of recent requests for live stats.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._lock = threading.Lock()

        # Rolling window of completed requests
        self._completed: deque[RequestMetrics] = deque(maxlen=window_size)

        # Counters
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._start_time = time.time()

        # Live state
        self.num_waiting = 0
        self.num_running = 0

    def on_request_arrival(self, request_id: str, prompt_tokens: int) -> RequestMetrics:
        with self._lock:
            self.total_requests += 1
            self.total_prompt_tokens += prompt_tokens
            self.num_waiting += 1

        return RequestMetrics(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            arrival_time=time.time(),
        )

    def on_request_started(self):
        with self._lock:
            self.num_waiting = max(0, self.num_waiting - 1)
            self.num_running += 1

    def on_token_generated(self, metrics: RequestMetrics):
        metrics.token_times.append(time.time())
        metrics.completion_tokens += 1

        if metrics.first_token_time is None:
            metrics.first_token_time = time.time()

        with self._lock:
            self.total_completion_tokens += 1

    def on_request_finished(self, metrics: RequestMetrics):
        metrics.finish_time = time.time()

        with self._lock:
            self.num_running = max(0, self.num_running - 1)
            self._completed.append(metrics)

    def get_stats(self) -> dict:
        """Get current system-wide statistics."""
        with self._lock:
            elapsed = time.time() - self._start_time

            # Compute averages over completed requests in the window
            ttft_values = [m.ttft for m in self._completed if m.ttft is not None]
            tbt_values = [m.tbt_avg for m in self._completed if m.tbt_avg is not None]
            throughputs = [
                m.generation_throughput for m in self._completed
                if m.generation_throughput is not None
            ]

            stats = {
                "uptime_seconds": round(elapsed, 1),
                "total_requests": self.total_requests,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "num_waiting": self.num_waiting,
                "num_running": self.num_running,
                "overall_throughput_tps": round(
                    (self.total_prompt_tokens + self.total_completion_tokens) / max(elapsed, 0.001), 1
                ),
            }

            if ttft_values:
                ttft_sorted = sorted(ttft_values)
                stats["ttft_avg_ms"] = round(sum(ttft_values) / len(ttft_values) * 1000, 1)
                stats["ttft_p50_ms"] = round(ttft_sorted[len(ttft_sorted) // 2] * 1000, 1)
                stats["ttft_p99_ms"] = round(
                    ttft_sorted[min(int(len(ttft_sorted) * 0.99), len(ttft_sorted) - 1)] * 1000, 1
                )

            if tbt_values:
                tbt_sorted = sorted(tbt_values)
                stats["tbt_avg_ms"] = round(sum(tbt_values) / len(tbt_values) * 1000, 1)
                stats["tbt_p50_ms"] = round(tbt_sorted[len(tbt_sorted) // 2] * 1000, 1)
                stats["tbt_p99_ms"] = round(
                    tbt_sorted[min(int(len(tbt_sorted) * 0.99), len(tbt_sorted) - 1)] * 1000, 1
                )

            if throughputs:
                stats["avg_request_throughput_tps"] = round(
                    sum(throughputs) / len(throughputs), 1
                )

            return stats

    def reset(self):
        with self._lock:
            self._completed.clear()
            self.total_requests = 0
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self._start_time = time.time()
            self.num_waiting = 0
            self.num_running = 0
