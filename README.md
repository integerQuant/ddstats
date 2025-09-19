# ddstats

Fast drawdown & CED metrics in Rust with NumPy bindings.

## Overview

`ddstats` provides high-performance financial metrics, including drawdown and Expected Drawdown (CED), implemented in Rust and exposed to Python via NumPy bindings. This allows for fast computations directly from Python, leveraging Rust's speed and safety.

## Features

- **Drawdown metrics**: Compute maximum drawdown and related statistics.
- **CED (Conditional Expected Drawdown)**: Efficient calculation for risk analysis.
- **NumPy integration**: Seamless usage from Python with NumPy arrays.
- **Cross-platform**: Works on Windows, macOS, and Linux.

## Installation

You can install `ddstats` using pip:

```sh
pip install ddstats
```

Or build from source:

```sh
git clone https://github.com/integerQuant/ddstats.git
cd ddstats
pip install maturin
maturin develop
```

## Usage

```python
import numpy as np
import ddstats

x = np.array([0.01, -0.02, 0.03], dtype=float)
max_dd = ddstats.max_drawdown(x)
print("Max Drawdown:", max_dd)
```

## API

### `ddstats.max_drawdown(x: np.ndarray) -> float`

Computes the maximum drawdown of a time series.

- **x**: 1D NumPy array of floats.

### `ddstats.ced(x: np.ndarray, level: float) -> float`

Computes the Conditional Expected Drawdown at a given confidence level.

- **x**: 1D NumPy array of floats.
- **level**: Confidence level (e.g., 0.95).

## Building

This project uses [maturin](https://github.com/PyO3/maturin) for building Python bindings:

```sh
maturin build
```

## License

MIT License. See [LICENSE](LICENSE).
