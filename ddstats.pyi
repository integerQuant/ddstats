from __future__ import annotations
from typing import Literal
from numpy.typing import NDArray
import numpy as np

def max_drawdown(returns: NDArray[np.float64], /) -> float:
    """
    Compute the maximum drawdown (MDD).

    Parameters
    ----------
    returns : numpy.ndarray (float64, 1-D)
        Return series over time.

    Returns
    -------
    float
        Positive drawdown fraction (e.g., 0.25 == -25%).
    """

def rolling_max_drawdown(
    returns: NDArray[np.float64],
    window: int,
    min_window: int | None = ...,
    step: int | None = ...,
    *,
    parallel: bool = True,
) -> NDArray[np.float64]:
    """
    Rolling maximum drawdown (MDD).

    Parameters
    ----------
    returns : ndarray[float64], shape (n,)
    window : int
    min_window : int, optional
    step : int, optional
    parallel : bool, default True

    Returns
    -------
    ndarray[float64]
        Rolling MDD values.
    """

def ced(
    returns: NDArray[np.float64],
    t: int = 21,
    alpha: float = 0.9,
    *,
    parallel: bool = True,
) -> float:
    """
    Conditional Expected Drawdown (CED).

    Parameters
    ----------
    returns : ndarray[float64], shape (n,)
    t : int, default 21
    alpha : float, default 0.9
    parallel : bool, default True

    Returns
    -------
    float
        Mean of the top (1 - alpha) tail of rolling MDDs (NumPy-linear, >= threshold).
    """

def expanding_ced(
    returns: NDArray[np.float64],
    t: int = 21,
    alpha: float = 0.9,
    *,
    method: Literal["heap", "sort"] = "heap",
    parallel: bool = True,
) -> NDArray[np.float64]:
    """
    Expanding CED series.

    Parameters
    ----------
    returns : ndarray[float64], shape (n,)
    t : int, default 21
    alpha : float, default 0.9
    method : {'heap','sort'}, default 'heap'
        'heap' matches NumPy selection (>= quantile) with O(n log n);
        'sort' is exact but slower.
    parallel : bool, default True

    Returns
    -------
    ndarray[float64]
        Expanding CED with NaN until index t-1.
    """
