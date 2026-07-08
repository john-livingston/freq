import numpy as np

from freq.util import get_alias, ordered_set


def test_get_alias_both_branches():
    plus, minus = get_alias(2.0, 1.0)
    assert np.isclose(plus, 1.0/1.5)
    assert np.isclose(minus, 2.0)


def test_get_alias_equal_periods_is_inf():
    plus, minus = get_alias(1.0, 1.0)
    assert np.isclose(plus, 0.5)
    assert np.isinf(minus)


def test_ordered_set_preserves_first_appearance():
    assert list(ordered_set(np.array(['b', 'a', 'b', 'c']))) == ['b', 'a', 'c']
