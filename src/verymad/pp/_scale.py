import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .._utils import CONSTANT


def _validate_input(x: np.ndarray | pd.Series | pd.DataFrame) -> tuple[np.ndarray, type, object]:
    if isinstance(x, pd.DataFrame):
        if not all(is_numeric_dtype(dtype) and not is_bool_dtype(dtype) for dtype in x.dtypes):
            raise TypeError("All DataFrame columns must be numeric.")
        return x.to_numpy(dtype=float, na_value=np.nan), pd.DataFrame, x
    if isinstance(x, pd.Series):
        if not is_numeric_dtype(x.dtype) or is_bool_dtype(x.dtype):
            raise TypeError("`x` must be numeric.")
        return x.to_numpy(dtype=float, na_value=np.nan), pd.Series, x
    array = np.asarray(x)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("`x` must be a numeric array, Series, or DataFrame.")
    if array.ndim not in {1, 2}:
        raise ValueError("`x` must be one- or two-dimensional.")
    return array.astype(float, copy=True), np.ndarray, x


def _median_and_mad(
    values: np.ndarray, axis: int, na_rm: bool, constant: float
) -> tuple[np.ndarray, np.ndarray]:
    reducer = np.nanmedian if na_rm else np.median
    centre = reducer(values, axis=axis)
    spread = constant * reducer(np.abs(values - np.expand_dims(centre, axis)), axis=axis)
    return centre, spread


def _restore(
    values: np.ndarray, kind: type, original: object
) -> np.ndarray | pd.Series | pd.DataFrame:
    if kind is pd.DataFrame:
        assert isinstance(original, pd.DataFrame)
        return pd.DataFrame(values, index=original.index, columns=original.columns)
    if kind is pd.Series:
        assert isinstance(original, pd.Series)
        return pd.Series(values, index=original.index, name=original.name)
    return values


def mad_scale(
    x: np.ndarray | pd.Series | pd.DataFrame,
    *,
    axis: int = 0,
    center: bool = True,
    scale: bool = True,
    constant: float = CONSTANT,
    na_rm: bool = True,
    zero_mad: str = "zero",
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Robustly center and/or scale a numeric vector or matrix by its MAD."""
    if axis not in {0, 1}:
        raise ValueError("`axis` must be 0 or 1.")
    if (
        not isinstance(center, (bool, np.bool_))
        or not isinstance(scale, (bool, np.bool_))
        or not isinstance(na_rm, (bool, np.bool_))
    ):
        raise TypeError("`center`, `scale`, and `na_rm` must be boolean.")
    if zero_mad not in {"zero", "na", "error"}:
        raise ValueError("`zero_mad` must be `zero`, `na`, or `error`.")
    if (
        isinstance(constant, (bool, np.bool_))
        or not isinstance(constant, (int, float, np.integer, np.floating))
        or not np.isfinite(constant)
        or constant <= 0
    ):
        raise ValueError("`constant` must be a positive finite number.")
    values, kind, original = _validate_input(x)
    if np.isinf(values).any():
        raise ValueError("`x` must not contain Inf or -Inf.")
    if values.size == 0:
        return _restore(values, kind, original)
    if values.ndim == 1:
        if axis != 0:
            raise ValueError("`axis=1` is only valid for two-dimensional input.")
        reducer = np.nanmedian if na_rm else np.median
        centre = reducer(values)
        spread = float(constant) * reducer(np.abs(values - centre))
        if scale and spread == 0:
            if zero_mad == "error":
                raise ValueError("MAD is zero.")
            safe_spread = 1.0
        else:
            safe_spread = spread
        result = values - centre if center else values.copy()
        if scale:
            result = result / safe_spread
        result[np.isnan(values)] = np.nan
        if scale and spread == 0 and zero_mad == "na":
            result[:] = np.nan
        return _restore(result, kind, original)

    centre, spread = _median_and_mad(values, axis, na_rm, float(constant))
    if scale and np.any(spread == 0):
        if zero_mad == "error":
            location = "row" if axis == 1 else "column"
            affected = int(np.flatnonzero(spread == 0)[0])
            raise ValueError(f"MAD is zero for {location} {affected}.")
        safe_spread = np.where(spread == 0, 1.0, spread)
    else:
        safe_spread = spread
    expanded_centre = np.expand_dims(centre, axis)
    expanded_spread = np.expand_dims(safe_spread, axis)
    result = values - expanded_centre if center else values.copy()
    if scale:
        result = result / expanded_spread
    result[np.isnan(values)] = np.nan
    if scale and zero_mad == "na":
        mask = np.expand_dims(spread == 0, axis)
        result = np.where(mask, np.nan, result)
        result[np.isnan(values)] = np.nan
    return _restore(result, kind, original)
