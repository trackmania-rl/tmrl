from tmrl.networking import Buffer


def test_buffer_append_respects_maxlen():
    b = Buffer(maxlen=3)
    for i in range(5):
        b.append_sample((i, i, 0.0, False, False, {}))
    assert len(b) == 3
    assert b.memory[0][0] == 2
    assert b.memory[-1][0] == 4


def test_buffer_clear_keeps_stats():
    b = Buffer(maxlen=10)
    b.stat_train_return = 1.5
    b.append_sample((0, 0, 0.0, False, False, {}))
    b.clear()
    assert len(b) == 0
    assert b.stat_train_return == 1.5


def test_buffer_iadd_merges_and_clips():
    a = Buffer(maxlen=4)
    b = Buffer(maxlen=4)
    for i in range(3):
        a.append_sample((i, i, float(i), False, False, {}))
    for i in range(3, 6):
        b.append_sample((i, i, float(i), False, False, {}))
    b.stat_train_return = 99.0
    a += b
    assert len(a) == 4
    assert a.stat_train_return == 99.0
