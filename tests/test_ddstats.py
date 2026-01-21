import importlib.metadata
import math

import numpy as np

import ddstats


def max_drawdown_py(returns: np.ndarray) -> float:
    if returns.size == 0 or np.isnan(returns).any():
        return math.nan
    cur_acc = 1.0
    cur_max = 1.0
    max_dd = 0.0
    for r in returns:
        cur_acc *= 1.0 + float(r)
        if cur_acc > cur_max:
            cur_max = cur_acc
        else:
            drawdown = (cur_acc - cur_max) / cur_max
            if drawdown < max_dd:
                max_dd = drawdown
    return -max_dd


def rolling_bounds_py(n: int, window: int, min_window: int, step: int) -> list[tuple[int, int]]:
    if n == 0 or min_window == 0 or step == 0 or n < min_window:
        return []
    bounds = []
    max_window_i = n - min_window + 1
    for i in range(0, max_window_i, step):
        if i < max(window - min_window, 0):
            i_window = min_window + i
            start_i = 0
        else:
            i_window = window
            start_i = i - (window - min_window)
        end_i = start_i + i_window
        if end_i > n:
            break
        bounds.append((start_i, end_i))
    return bounds


def test_max_drawdown_basic_cases() -> None:
    data = np.array([0.10, -0.10], dtype=float)
    assert math.isclose(ddstats.max_drawdown(data), 0.10, rel_tol=0.0, abs_tol=1e-12)
    data_up = np.array([0.01, 0.02, 0.03], dtype=float)
    assert ddstats.max_drawdown(data_up) == 0.0
    data_nan = np.array([0.01, np.nan], dtype=float)
    assert math.isnan(ddstats.max_drawdown(data_nan))


def test_rolling_max_drawdown_matches_python() -> None:
    rets = np.array([0.01, -0.02, 0.03, -0.01, 0.02], dtype=float)
    expected = []
    for start, end in rolling_bounds_py(len(rets), 3, 2, 1):
        expected.append(max_drawdown_py(rets[start:end]))
    out = ddstats.rolling_max_drawdown(rets, window=3, min_window=2, step=1, parallel=False)
    np.testing.assert_allclose(out, np.array(expected), rtol=0.0, atol=1e-12, equal_nan=True)


def test_ced_matches_python_reference() -> None:
    rets = np.array([0.01, -0.02, 0.03, -0.01, 0.02], dtype=float)
    t = 3
    rolling = []
    for start, end in rolling_bounds_py(len(rets), t, t, 1):
        rolling.append(max_drawdown_py(rets[start:end]))
    rolling = np.array(rolling)
    q = np.quantile(rolling, 0.9, method="linear")
    expected = rolling[rolling >= q].mean()
    out = ddstats.ced(rets, t=t, alpha=0.9, parallel=False)
    assert math.isclose(out, expected, rel_tol=0.0, abs_tol=1e-12)


def test_expanding_ced_heap_matches_sort() -> None:
    rets = np.array([0.01, -0.02, 0.03, -0.01, 0.02], dtype=float)
    heap = ddstats.expanding_ced(rets, t=3, alpha=0.9, method="heap", parallel=False)
    sort = ddstats.expanding_ced(rets, t=3, alpha=0.9, method="sort", parallel=False)
    np.testing.assert_allclose(heap, sort, rtol=0.0, atol=1e-12, equal_nan=True)


def test_version_matches_metadata() -> None:
    assert ddstats.__version__ == importlib.metadata.version("ddstats")
