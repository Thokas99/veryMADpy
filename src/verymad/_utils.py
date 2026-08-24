from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

CONSTANT = 1.4826
TRANSFORMS = {"none", "log1p", "log10"}


def validate_metrics(metrics: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(metrics, Mapping) or not metrics:
        raise TypeError("`metrics` must be a non-empty mapping of column names to directions.")
    result: dict[str, str] = {}
    for metric, direction in metrics.items():
        if not isinstance(metric, str) or not metric:
            raise ValueError("Metric names must be non-empty strings.")
        if direction not in {"lower", "upper", "both"}:
            raise ValueError("Metric directions must be `lower`, `upper`, or `both`.")
        result[metric] = direction
    return result


def resolve_transforms(transform: str | Mapping[str, str], metrics: list[str]) -> dict[str, str]:
    if isinstance(transform, str):
        if transform not in TRANSFORMS:
            raise ValueError("Transformations must be `none`, `log1p`, or `log10`.")
        return dict.fromkeys(metrics, transform)
    if not isinstance(transform, Mapping) or not transform:
        raise TypeError("`transform` must be a supported string or non-empty mapping.")
    unknown = set(transform) - set(metrics)
    if unknown:
        raise ValueError(
            f"Transformation names must reference selected metrics: {sorted(unknown)}."
        )
    result = dict.fromkeys(metrics, "none")
    for metric, method in transform.items():
        if not isinstance(method, str) or method not in TRANSFORMS:
            raise ValueError("Transformations must be `none`, `log1p`, or `log10`.")
        result[metric] = method
    return result


def validate_positive(value: Real, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"`{name}` must be a positive finite number.")
    return float(value)


def validate_min_n(value: Real) -> int:
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0 or value != int(value):
        raise ValueError("`min_n` must be a positive integer.")
    return int(value)


def numeric_values(values: pd.Series, metric: str) -> np.ndarray:
    if not is_numeric_dtype(values) or is_bool_dtype(values):
        raise TypeError(f"QC metric column(s) must be numeric: {metric}.")
    array = values.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(array).any():
        raise ValueError(f"Metric `{metric}` contains non-finite values.")
    return array


def transform_values(values: np.ndarray, method: str, metric: str) -> np.ndarray:
    present = values[np.isfinite(values)]
    if method == "log1p" and (present < 0).any():
        raise ValueError(f"`log1p` requires non-negative values in `{metric}`.")
    if method == "log10" and (present <= 0).any():
        raise ValueError(f"`log10` requires positive values in `{metric}`.")
    if method == "log1p":
        return np.log1p(values)
    if method == "log10":
        return np.log10(values)
    return values


def mad(values: np.ndarray) -> tuple[float, float]:
    centre = float(np.nanmedian(values))
    spread = CONSTANT * float(np.nanmedian(np.abs(values - centre)))
    return centre, spread


def inverse_transform(value: float | None, method: str) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    if method == "log1p":
        return float(np.expm1(value))
    if method == "log10":
        return float(10**value)
    return float(value)
