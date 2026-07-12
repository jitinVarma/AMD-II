"""Unit tests for agent.frames adaptive frame counting and timestamp
selection. Pure arithmetic/logic -- no network, no ffmpeg, no remote clips.
"""
import unittest

from agent.frames import adaptive_frame_count, _select_timestamps


class AdaptiveFrameCountTests(unittest.TestCase):
    # (duration_seconds, expected_count) -- ceil(duration / 7.0) clamped to
    # [6, 14].
    CASES = [
        (3, 6),
        (7, 6),
        (14, 6),
        (30, 6),
        (42, 6),    # ceil(6.0) = 6 -- exactly at the floor boundary
        (43, 7),    # ceil(6.14) = 7 -- just past the floor boundary
        (60, 9),
        (98, 14),   # ceil(14.0) = 14 -- exactly at the cap
        (180, 14),  # ceil(25.71) = 26 -- clamped to the cap
    ]

    def test_expected_counts(self):
        for duration, expected in self.CASES:
            with self.subTest(duration=duration):
                self.assertEqual(adaptive_frame_count(duration, override=None), expected)

    def test_floor_never_violated(self):
        for duration in [0.1, 1, 3, 6.9, 7, 7.1]:
            with self.subTest(duration=duration):
                self.assertGreaterEqual(adaptive_frame_count(duration, override=None), 6)

    def test_cap_never_exceeded(self):
        for duration in [98, 150, 300, 600, 3600]:
            with self.subTest(duration=duration):
                self.assertLessEqual(adaptive_frame_count(duration, override=None), 14)

    def test_override_disables_adaptation(self):
        # override wins regardless of duration, subject only to the
        # existing max(1, override) minimum-valid-value behavior.
        self.assertEqual(adaptive_frame_count(3, override=10), 10)
        self.assertEqual(adaptive_frame_count(300, override=2), 2)
        self.assertEqual(adaptive_frame_count(60, override=0), 1)


class SelectTimestampsShortClipTests(unittest.TestCase):
    def test_three_second_clip_produces_six_valid_distinct_sorted_timestamps(self):
        duration = 3.0
        timestamps = _select_timestamps(duration, target_count=6, scene_changes=[])

        self.assertEqual(len(timestamps), 6, "expected exactly six timestamps")
        self.assertEqual(len(set(timestamps)), 6, "timestamps must all be distinct, no duplicates")
        self.assertEqual(timestamps, sorted(timestamps), "timestamps must be in chronological order")
        for t in timestamps:
            self.assertGreaterEqual(t, 0.0, f"timestamp {t} is before the start of the clip")
            self.assertLessEqual(t, duration, f"timestamp {t} is beyond the clip duration")


if __name__ == "__main__":
    unittest.main()
