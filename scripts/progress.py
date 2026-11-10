"""Lightweight, monotonic, phase-local ETA and heartbeat logging (no ML changes)."""
from __future__ import annotations

import math
import threading
import time

from .common import log


def duration(seconds: float) -> str:
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class PhaseProgress:
    """ETA covers ONLY this phase, not unmeasured downstream work.

    Cumulative throughput naturally includes decoding, tile/TTA inference and
    warm-up. During an unusually slow current item we show recalculating, rather
    than counting down to a misleading zero. Unknown-length phases get heartbeats
    but no invented ETA. The optional clock and disabled heartbeat support tests.
    """

    def __init__(self, stage, label, total=None, *, interval=15., clock=None, sink=None):
        if total is not None and (not isinstance(total, int) or total < 0):
            raise ValueError("total must be a nonnegative integer or None")
        self.stage, self.label, self.total = stage, label, total
        self.interval = interval
        self.clock = clock or time.monotonic
        self.sink = sink or log
        self.started = self.clock()
        self.last_completed_at = self.started
        self.last_log_at = self.started
        self.completed = 0
        self.detail = ""
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.report()
        if self.interval > 0:
            self.thread = threading.Thread(target=self._heartbeat, daemon=True,
                                           name=f"progress-{self.stage}")
            self.thread.start()
        return self

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.report()

    def message(self, status=None):
        with self.lock:
            now = self.clock()
            elapsed = max(0., now - self.started)
            done, total = self.completed, self.total
            prefix = self.label
            if total is not None:
                percent = 100. * done / total if total else 100.
                prefix += f" {done}/{total} ({percent:.1f}%)"
            parts = [prefix, f"elapsed {duration(elapsed)}"]
            if status is not None:
                parts.append(status)
            elif total is None:
                parts.append("phase ETA unavailable (variable work)")
            elif done == total:
                parts.append("phase ETA ~00:00:00")
            elif done < 3 or self.last_completed_at <= self.started:
                parts.append("phase ETA estimating (warming up)")
            else:
                average = (self.last_completed_at - self.started) / done
                pending = max(0., now - self.last_completed_at)
                if pending > 2. * average:
                    parts.append("phase ETA recalculating (current item slower than average)")
                else:
                    remaining = max(1., average * (total - done) - pending)
                    parts.extend([f"phase ETA ~{duration(remaining)}", f"{average:.2f} s/item"])
            if self.detail:
                parts.append(self.detail)
            return "; ".join(parts)

    def report(self):
        self.sink(self.stage, self.message())
        with self.lock:
            self.last_log_at = self.clock()

    def update(self, completed, detail=""):
        with self.lock:
            if (not isinstance(completed, int) or completed < self.completed or
                    (self.total is not None and completed > self.total)):
                raise ValueError("completed count must increase within the declared total")
            advanced = completed > self.completed
            self.completed = completed
            if advanced:
                self.last_completed_at = self.clock()
            self.detail = detail
            emit = advanced and (completed in (1, 3, self.total) or
                                self.clock() - self.last_log_at >= max(1., self.interval))
        if emit:
            self.report()

    def finish(self, status=None):
        if self.stop.is_set():
            return
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.)
        if status is None:
            status = "complete" if self.total is None or self.completed == self.total else "stopped before completion"
        self.sink(self.stage, self.message(status))

    def __exit__(self, exc_type, exc, tb):
        self.finish("failed/interrupted" if exc_type else None)
        return False
