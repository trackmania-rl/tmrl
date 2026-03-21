"""Tests for Buffer thread-safety, append/overflow, iadd, clear, speed bonus."""

import threading

from tmrl.networking import Buffer


def _make_sample(rew=1.0, terminated=False, truncated=False):
    return (None, None, rew, terminated, truncated, {"reward_sum": rew})


class TestBufferBasic:
    def test_append_and_len(self):
        buf = Buffer(maxlen=10)
        for i in range(5):
            buf.append_sample(_make_sample(rew=float(i)))
        assert len(buf) == 5

    def test_overflow_clips(self):
        buf = Buffer(maxlen=3)
        for i in range(10):
            buf.append_sample(_make_sample(rew=float(i)))
        assert len(buf) == 3
        rewards = [s[2] for s in buf.memory]
        assert rewards == [7.0, 8.0, 9.0]

    def test_clear_empties(self):
        buf = Buffer(maxlen=10)
        buf.append_sample(_make_sample())
        buf.clear()
        assert len(buf) == 0

    def test_iadd_merges(self):
        b1 = Buffer(maxlen=100)
        b2 = Buffer(maxlen=100)
        for i in range(3):
            b1.append_sample(_make_sample(rew=float(i)))
        for i in range(4):
            b2.append_sample(_make_sample(rew=float(i + 10)))
        b2.stat_train_return = 42.0
        b1 += b2
        assert len(b1) == 7
        assert b1.stat_train_return == 42.0


class TestBufferSpeedBonus:
    def test_noop_when_empty(self):
        buf = Buffer(maxlen=10)
        buf.apply_speed_bonus(1.0)
        assert len(buf) == 0

    def test_noop_when_zero_scale(self):
        buf = Buffer(maxlen=10)
        buf.append_sample(_make_sample(rew=5.0))
        buf.apply_speed_bonus(0.0)
        assert buf.memory[0][2] == 5.0

    def test_bonus_applied(self):
        buf = Buffer(maxlen=10)
        n = 4
        for _ in range(n):
            buf.append_sample(_make_sample(rew=1.0))
        buf.apply_speed_bonus(16.0)
        bonus_per_step = 16.0 / (n * n)
        expected_rew = 1.0 + bonus_per_step
        for s in buf.memory:
            assert abs(s[2] - expected_rew) < 1e-6


class TestBufferThreadSafe:
    def test_lock_created_when_thread_safe(self):
        buf = Buffer(maxlen=10, thread_safe=True)
        assert buf._lock is not None

    def test_no_lock_by_default(self):
        buf = Buffer(maxlen=10)
        assert buf._lock is None

    def test_concurrent_appends(self):
        buf = Buffer(maxlen=10000, thread_safe=True)
        n_per_thread = 500
        n_threads = 4

        def worker():
            for i in range(n_per_thread):
                buf.append_sample(_make_sample(rew=float(i)))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(buf) == n_per_thread * n_threads
