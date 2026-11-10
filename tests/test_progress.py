import threading
import unittest

from scripts.progress import PhaseProgress, duration


class ProgressTests(unittest.TestCase):
    def make_progress(self, total=10):
        now = [0.]
        lines = []
        progress = PhaseProgress("test", "frames", total, interval=0,
                                 clock=lambda: now[0], sink=lambda stage, line: lines.append(line))
        return progress, now, lines

    def test_eta_after_warmup_and_completion(self):
        progress, now, lines = self.make_progress()
        with progress:
            self.assertIn("estimating", lines[-1])
            for frame in (1, 2, 3):
                now[0] = frame * 10.
                progress.update(frame)
            self.assertIn("phase ETA ~00:01:10", progress.message())
            self.assertIn("elapsed 00:00:30", progress.message())
            now[0] = 100.
            progress.update(10)
            self.assertIn("phase ETA ~00:00:00", progress.message())
        self.assertIn("complete", lines[-1])

    def test_slow_frame_recalculates_instead_of_fake_zero(self):
        progress, now, _ = self.make_progress()
        now[0] = 30.
        progress.update(3)
        now[0] = 1000.
        self.assertIn("recalculating", progress.message())
        self.assertNotIn("phase ETA ~00:00:00", progress.message())

    def test_unknown_length_does_not_invent_eta(self):
        progress, now, _ = self.make_progress(None)
        now[0] = 3700.
        self.assertIn("elapsed 01:01:40", progress.message())
        self.assertIn("ETA unavailable", progress.message())

    def test_fast_cadence_overhead_is_included(self):
        progress, now, _ = self.make_progress(8)
        for completed, elapsed in ((1, 10.), (2, 10.1), (3, 20.1), (4, 20.2)):
            now[0] = elapsed
            progress.update(completed)
        self.assertIn("phase ETA ~00:00:21", progress.message())
