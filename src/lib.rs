//! Utilities for drawdown analytics and tail-risk statistics on return series.
//!
//! This module provides:
//! - [`max_drawdown_core`] – maximum drawdown of a return slice.
//! - Rolling window helpers ([`rolling_bounds`], `rolling_*` variants).
//! - A linear-interpolated quantile helper ([`quantile_linear`]) matching NumPy.
//! - Expanding Conditional Expected Drawdown (CED) computed two ways:
//!   - [`expanding_ced_heap_core`] (online, heap-based; efficient).
//!   - [`expanding_ced_sort_core`] (sorting-based; simpler but slower).
//!
//! # Definitions
//! - **Return series** `rets[i]` is interpreted as *simple* returns (e.g., `0.01` = +1%).
//! - **Cumulative wealth**: `acc_t = ∏_{i=0..t} (1 + rets[i])` starting at 1.0.
//! - **Drawdown at t**: `(acc_t - max_{τ≤t} acc_τ) / max_{τ≤t} acc_τ ≤ 0`.
//! - **Max drawdown (MDD)** over an interval is the minimum drawdown (most negative), reported as a **non-negative** number here.
//! - **CED_α** over a prefix is the average of the worst `α` tail of past MDDs (see details on `alpha` below).
//!
//! # Conventions & Edge Cases
//! - Any `NaN` in inputs yields `NaN` at the corresponding aggregate position.
//! - Empty inputs return `NaN`.
//! - All functions are **panic-free** under documented preconditions. `NotNan::new` is only fed non-NaN values.
//!
//! # Parallelism
//! Functions with a `parallel: bool` flag can compute rolling MDDs via Rayon when `true`.
//!
//! # Complexity (informal)
//! - [`max_drawdown_core`]: `O(n)` time, `O(1)` space.
//! - Rolling MDD helpers: `O(n)` windows × `O(window)` each (no reuse); parallel version distributes windows.
//! - [`expanding_ced_heap_core`]: overall ~`O(n log n)` due to heap maintenance across prefixes.
//! - [`expanding_ced_sort_core`]: ~`O(n^2 log n)` (re-sorts for each prefix); mainly for validation.
//!
//! # Notes on `alpha` (tail fraction)
//! We use NumPy's tail-counting convention for expanding CED at prefix length `L`:
//! `k = L - ceil(alpha * (L - 1))`, then average the **k largest** MDDs up to that point, with "tie promotion" to include all values equal to the threshold.
//!
//! # Examples
//! The items here are private; examples are marked `ignore` to avoid doctest compilation errors.
//!
//! ```rust,ignore
//! // Maximum drawdown on a simple path: up 10%, down 10%.
//! let rets = [0.10, -0.10];
//! let mdd = max_drawdown_core(&rets);
//! assert!(mdd >= 0.0);
//!
//! // Rolling bounds for a 5-length series, window=3, min_window=2, step=1.
//! let b = rolling_bounds(5, 3, 2, 1);
//! // b yields start/end (exclusive) pairs such as [(0,2),(0,3),(1,4),(2,5)] depending on params.
//! ```
//!

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ordered_float::NotNan;
use pyo3::prelude::*;
use pyo3::{types::PyModule, Bound};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Computes the **maximum drawdown** (MDD) of a return slice.
///
/// The return series is interpreted as *simple* returns (`r_t`), and the
/// cumulative wealth path is `acc_t = ∏ (1 + r_t)` with `acc_0 = 1`.
/// The maximum drawdown is returned as a **non-negative** number.
///
/// # Arguments
/// - `rets`: slice of simple returns (e.g., `0.01` = +1%).
///
/// # Returns
/// - Non-negative `f64` equal to the magnitude of the worst drawdown.
/// - `NaN` if `rets` is empty or contains any `NaN`.
///
/// # Complexity
/// - Time: `O(n)`
/// - Space: `O(1)`
///
/// # Examples
/// ```rust,ignore
/// let mdd = max_drawdown_core(&[0.02, 0.03, -0.10, 0.01]);
/// assert!(mdd >= 0.0);
/// ```
#[inline(always)]
fn max_drawdown_core(rets: &[f64]) -> f64 {
    let n = rets.len();
    if n == 0 { return f64::NAN; }
    if rets.iter().any(|x| x.is_nan()) { return f64::NAN; }

    let mut cur_acc = 1.0_f64;
    let mut cur_max = 1.0_f64;
    let mut max_dd = 0.0_f64;

    for &r in rets {
        cur_acc *= 1.0 + r;
        if cur_acc > cur_max {
            cur_max = cur_acc;
        } else {
            let drawdown = (cur_acc - cur_max) / cur_max; // ≤ 0
            if drawdown < max_dd {
                max_dd = drawdown;
            }
        }
    }
    -max_dd
}

/// Produces `(start, end)` index pairs for rolling windows over a series,
/// supporting a minimum warm-up and stride.
///
/// The pairs use **half-open** ranges `[start, end)` and are guaranteed to
/// satisfy `end - start >= min_window` and `end <= n`.
///
/// # Arguments
/// - `n`: total length of the series.
/// - `window`: target window size once warm-up is complete.
/// - `min_window`: minimum window size for the earliest windows (warm-up).
/// - `step`: stride between consecutive windows (≥ 1).
///
/// # Returns
/// - Vector of `(start, end)` pairs. Empty if any of:
///   - `n == 0`, `min_window == 0`, `step == 0`, or `n < min_window`.
///
/// # Behavior
/// For early windows where the full `window` is not yet available, the function
/// grows from `min_window` up to `window`, keeping `start = 0` until possible.
/// Thereafter, it slides the fixed `window` forward by `step`.
///
/// # Examples
/// ```rust,ignore
/// // For n=5, window=3, min_window=2, step=1:
/// // yields: [(0,2), (0,3), (1,4), (2,5)]
/// let b = rolling_bounds(5, 3, 2, 1);
/// ```
#[inline]
fn rolling_bounds(n: usize, window: usize, min_window: usize, step: usize) -> Vec<(usize, usize)> {
    if n == 0 || min_window == 0 || step == 0 || n < min_window {
        return Vec::new();
    }
    let max_window_i = n - min_window + 1;
    let mut bounds = Vec::with_capacity((max_window_i + step - 1) / step);
    for i in (0..max_window_i).step_by(step) {
        let (i_window, start_i) = if i < window.saturating_sub(min_window) {
            (min_window + i, 0)
        } else {
            (window, i - (window - min_window))
        };
        let end_i = start_i + i_window;
        if end_i > n { break; }
        bounds.push((start_i, end_i));
    }
    bounds
}

/// Computes rolling **maximum drawdown** for all windows defined by
/// `window`, `min_window`, and `step` (see [`rolling_bounds`]).
///
/// # Arguments
/// - `rets`: return series.
/// - `window`: target window size after warm-up.
/// - `min_window`: minimum initial window size.
/// - `step`: stride between windows.
///
/// # Returns
/// - Vector of MDDs for each produced window. Empty if no windows.
///
/// # Notes
/// - Returns `NaN` for any window that contains a `NaN`.
fn rolling_max_drawdown_core(rets: &[f64], window: usize, min_window: usize, step: usize) -> Vec<f64> {
    let n = rets.len();
    let bounds = rolling_bounds(n, window, min_window, step);
    let mut out = Vec::with_capacity(bounds.len());
    for (s, e) in bounds { out.push(max_drawdown_core(&rets[s..e])); }
    out
}

/// As [`rolling_max_drawdown_core`] but computes windows **in parallel**
/// using Rayon when available.
///
/// # Parallelism
/// Each window is independent and mapped over `par_iter()`.
fn rolling_max_drawdown_core_par(rets: &[f64], window: usize, min_window: usize, step: usize) -> Vec<f64> {
    let n = rets.len();
    let bounds = rolling_bounds(n, window, min_window, step);
    bounds
        .par_iter()
        .map(|&(s, e)| max_drawdown_core(&rets[s..e]))
        .collect()
}

/// Linear-interpolated quantile matching NumPy's default behavior.
///
/// Given `alpha ∈ [0,1]`, computes the `alpha` quantile of `data` with linear
/// interpolation between order statistics:
/// - Sorts ascending.
/// - Let `k = alpha * (n - 1)`, `lo = floor(k)`, `hi = ceil(k)`.
/// - If `lo == hi`, returns `data[lo]`.
/// - Else returns `data[lo] + (data[hi] - data[lo]) * (k - lo)`.
///
/// # Arguments
/// - `data`: values to aggregate (copied and sorted internally).
/// - `alpha`: quantile in `[0, 1]`.
///
/// # Returns
/// - Quantile value, or `NaN` if `data` is empty, outside-range `alpha`, or any `NaN` present.
///
/// # Complexity
/// - Time: `O(n log n)` due to sorting.
/// - Space: `O(n)` (local copy).
#[inline]
fn quantile_linear(mut data: Vec<f64>, alpha: f64) -> f64 {
    let n = data.len();
    if n == 0 || !(0.0..=1.0).contains(&alpha) { return f64::NAN; }
    if data.iter().any(|x| x.is_nan()) { return f64::NAN; }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let k = alpha * (n as f64 - 1.0);
    let lo = k.floor() as usize;
    let hi = k.ceil() as usize;
    if lo == hi {
        data[lo]
    } else {
        let w = k - lo as f64;
        data[lo] + (data[hi] - data[lo]) * w
    }
}

/// Expanding **Conditional Expected Drawdown** (CED) using an online,
/// heap-based algorithm with tie promotion and NumPy-compatible tail counting.
///
/// For each index `a ≥ t-1`, we consider the list of rolling MDDs for every
/// trailing window of length `t` ending at or before `a` (i.e., all windows in
/// `rolling_max_drawdown_core(_ , t, t, 1)` up to that prefix). We then compute
/// the average of the **k largest** MDDs where
/// `k = L - ceil(alpha*(L-1))` and `L` is the number of available MDDs
/// up to that prefix. If multiple values equal the threshold, they are all
/// **promoted** into the average (tie promotion).
///
/// # Arguments
/// - `rets`: return series.
/// - `t`: fixed rolling window length for MDDs (must be `≥ 1`).
/// - `alpha`: tail fraction in `[0, 1]`. Larger `alpha` → **smaller** tail (more extreme).
/// - `parallel`: whether to parallelize the underlying rolling MDD computation.
///
/// # Returns
/// - A vector of length `n`, with `NaN` for indices `< t-1`.
/// - Also `NaN` at and after the first index whose MDDs contain any `NaN`.
///
/// # Complexity
/// - Rolling MDDs: `O(n * t)` naïvely (parallelizable across windows).
/// - Online CED maintenance: amortized `O(log n)` per step via two heaps.
///
/// # Examples
/// ```rust,ignore
/// let ced = expanding_ced_heap_core(&[0.01, -0.02, 0.03, -0.01], 3, 0.95, true);
/// assert_eq!(ced.len(), 4);
/// ```
fn expanding_ced_heap_core(rets: &[f64], t: usize, alpha: f64, parallel: bool) -> Vec<f64> {
    let n = rets.len();
    let mut out = vec![f64::NAN; n];
    if n < t { return out; }

    let mdds: Vec<f64> = if parallel {
        rolling_max_drawdown_core_par(rets, t, t, 1)
    } else {
        rolling_max_drawdown_core(rets, t, t, 1)
    };

    // Two-heaps structure:
    // - `upper`: min-heap (via Reverse) holding the current tail we average.
    // - `lower`: max-heap holding the rest.
    let mut lower: BinaryHeap<NotNan<f64>> = BinaryHeap::new();
    let mut upper: BinaryHeap<Reverse<NotNan<f64>>> = BinaryHeap::new();
    let mut sum_upper = 0.0_f64;
    let mut saw_nan = false;

    // Rebalance heaps so that upper.len() == k
    let rebalance_to_k = |k: usize,
                        lower: &mut BinaryHeap<NotNan<f64>>,
                        upper: &mut BinaryHeap<Reverse<NotNan<f64>>>,
                        sum_upper: &mut f64| {
        while upper.len() > k {
            let Reverse(v) = upper.pop().unwrap();
            *sum_upper -= v.into_inner();
            lower.push(v);
        }
        while upper.len() < k {
            if let Some(v) = lower.pop() {
                *sum_upper += v.into_inner();
                upper.push(Reverse(v));
            } else { break; }
        }
    };

    for a in (t - 1)..n {
        let idx = a - (t - 1);
        let len_prefix = idx + 1;

        let v = mdds[idx];
        if v.is_nan() { saw_nan = true; }
        if saw_nan { out[a] = f64::NAN; continue; }

        let nv = NotNan::new(v).unwrap();

        if upper.is_empty() {
            sum_upper += v;
            upper.push(Reverse(nv));
        } else if let Some(&Reverse(ref min_top)) = upper.peek() {
            if v > min_top.into_inner() {
                let Reverse(th) = upper.pop().unwrap();
                sum_upper -= th.into_inner();
                lower.push(th);
                sum_upper += v;
                upper.push(Reverse(nv));
            } else {
                lower.push(nv);
            }
        } else {
            sum_upper += v;
            upper.push(Reverse(nv));
        }

        // NumPy tail count: k = L - ceil(alpha*(L-1)), ensure k >= 1
        let hi = (alpha * ((len_prefix - 1) as f64)).ceil() as usize;
        let mut k = len_prefix.saturating_sub(hi);
        if k == 0 { k = 1; }

        rebalance_to_k(k, &mut lower, &mut upper, &mut sum_upper);

        // Tie promotion: move all values equal to the upper's min into upper.
        if let Some(&Reverse(ref min_top)) = upper.peek() {
            let thr = min_top.clone();
            while let Some(max_lower) = lower.peek() {
                if *max_lower == thr {
                    let v2 = lower.pop().unwrap();
                    sum_upper += v2.into_inner();
                    upper.push(Reverse(v2));
                } else { break; }
            }
        }

        out[a] = if upper.is_empty() { f64::NAN } else { sum_upper / (upper.len() as f64) };
    }

    out
}

/// Expanding **Conditional Expected Drawdown** (CED) via a simple,
/// sorting-based method. Useful for testing/validation against
/// [`expanding_ced_heap_core`].
///
/// # Arguments
/// - `rets`: return series.
/// - `t`: fixed rolling window length for MDDs.
/// - `alpha`: tail fraction in `[0,1]`.
/// - `parallel`: whether to parallelize the underlying rolling MDD computation.
///
/// # Returns
/// - A vector of length `n`, with `NaN` for indices `< t-1`.
///
/// # Notes
/// - For each prefix, computes the `alpha` quantile of all past MDDs using
///   [`quantile_linear`], then averages all MDDs `>=` that quantile (with
///   natural inclusion of ties).
///
/// # Complexity
/// - Potentially expensive: builds/sorts a growing slice at each step
///   (roughly `O(n^2 log n)`).
fn expanding_ced_sort_core(rets: &[f64], t: usize, alpha: f64, parallel: bool) -> Vec<f64> {
    let n = rets.len();
    let mut out = vec![f64::NAN; n];
    if n < t { return out; }
    let r = if parallel {
        rolling_max_drawdown_core_par(rets, t, t, 1)
    } else {
        rolling_max_drawdown_core(rets, t, t, 1)
    };
    for a in (t - 1)..n {
        let end = a - (t - 1) + 1;
        let slice = r[..end].to_vec();
        let q = quantile_linear(slice.clone(), alpha);
        if q.is_nan() { out[a] = f64::NAN; continue; }
        let (mut sum, mut cnt) = (0.0, 0usize);
        for &v in &slice {
            if v >= q { sum += v; cnt += 1; }
        }
        out[a] = if cnt == 0 { f64::NAN } else { sum / (cnt as f64) };
    }
    out
}


// ---------- Python bindings ----------

#[pyfunction]
#[pyo3(signature = (returns))]
fn max_drawdown(returns: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let rets = returns.as_slice()?;
    Ok(max_drawdown_core(rets))
}

#[pyfunction]
#[pyo3(signature = (returns, window, min_window=None, step=None, *, parallel=true))]
fn rolling_max_drawdown(
    py: Python<'_>,
    returns: PyReadonlyArray1<f64>,
    window: usize,
    min_window: Option<usize>,
    step: Option<usize>,
    parallel: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let minw = min_window.unwrap_or(window);
    let stp = step.unwrap_or(1);

    // SAFETY: copy data so it's independent of Python while we run without the GIL
    let rets: Vec<f64> = returns.as_slice()?.to_vec();

    // allow_threads -> detach (renamed)
    let v = py.detach(|| {
        if parallel {
            rolling_max_drawdown_core_par(&rets, window, minw, stp)
        } else {
            rolling_max_drawdown_core(&rets, window, minw, stp)
        }
    });

    let arr = v.into_pyarray(py);        
    Ok(arr.unbind())   
}

#[pyfunction]
#[pyo3(signature = (returns, t=21, alpha=0.9, *, parallel=true))]
fn ced(
    py: Python<'_>,
    returns: PyReadonlyArray1<f64>,
    t: usize,
    alpha: f64,
    parallel: bool,
) -> PyResult<f64> {
    let rets = returns.as_slice()?;
    let val = py.detach(|| {
        let r = if parallel {
            rolling_max_drawdown_core_par(rets, t, t, 1)
        } else {
            rolling_max_drawdown_core(rets, t, t, 1)
        };
        if r.is_empty() { return f64::NAN; }
        let q = quantile_linear(r.clone(), alpha);
        if q.is_nan() { return f64::NAN; }
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for &v in &r {
            if v >= q { sum += v; cnt += 1; }
        }
        if cnt == 0 { f64::NAN } else { sum / cnt as f64 }
    });
    Ok(val)
}

#[pyfunction]
#[pyo3(signature = (returns, t=21, alpha=0.9, *, method="heap", parallel=true))]
fn expanding_ced(
    py: Python<'_>,
    returns: PyReadonlyArray1<f64>,
    t: usize,
    alpha: f64,
    method: &str,
    parallel: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let rets = returns.as_slice()?;
    let v = py.detach(|| match method {
        "heap" => expanding_ced_heap_core(rets, t, alpha, parallel),
        "sort" => expanding_ced_sort_core(rets, t, alpha, parallel),
        _ => expanding_ced_heap_core(rets, t, alpha, parallel),
    });
    let arr = v.into_pyarray(py);        
    Ok(arr.unbind())   
}

#[pymodule]
fn ddstats(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__doc__", "ddstats: drawdown & CED metrics in Rust with NumPy bindings.")?;
    m.add("__version__", "0.4.0")?;
    m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(ced, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_ced, m)?)?;
    Ok(())
}
