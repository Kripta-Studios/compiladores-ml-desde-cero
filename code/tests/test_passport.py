import pytest
from lumbre.passport import Launch, verify


@pytest.mark.parametrize("shape", [(0, 7), (1, 1), (16, 16), (17, 23), (31, 65)])
def test_logical_map(shape):
    r = verify(Launch(*shape))
    assert r["status"] == "LOGICAL_MAP_VALIDATED"
    assert r["valid_writes"] == shape[0] * shape[1]


def test_wrong_stride_detected():
    r = verify(Launch(3, 5, 2, 4), lambda y, x, s: y * s.rows + x)
    assert r["status"] == "FAILED"
    assert r["missing"] and r["duplicate_writes"]


def test_outside_detected():
    r = verify(Launch(5, 3), lambda y, x, s: y * s.rows + x)
    assert r["out_of_range"]


def test_one_origin_bug():
    r = verify(Launch(2, 3), lambda y, x, s: y * s.columns + x + 1)
    assert r["missing"] == [0]
    assert r["out_of_range"] == [(1, 2, 6)]


def test_invalid_launch():
    for args in [(-1, 3), (2, 3, 0, 8), (1001, 1001)]:
        with pytest.raises(ValueError):
            Launch(*args)
    with pytest.raises(TypeError):
        Launch(2.5, 3)
