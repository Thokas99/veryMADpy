import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd

from .._utils import (
    CONSTANT,
    inverse_transform,
    mad,
    numeric_values,
    resolve_transforms,
    transform_values,
    validate_metrics,
    validate_min_n,
    validate_positive,
)


@dataclass(frozen=True)
class _MetricResult:
    flag: pd.Series
    threshold: dict[str, object]
    status: str


def _combine_flags(flags: list[pd.Series], index: pd.Index) -> pd.Series:
    if not flags:
        return pd.Series([], index=index, dtype="boolean")
    values = pd.DataFrame(
        {column: flag.to_numpy(dtype=object) for column, flag in enumerate(flags)},
        index=index,
        dtype="boolean",
    )
    any_true = values.fillna(False).any(axis=1)
    any_missing = values.isna().any(axis=1)
    combined = any_true.mask(~any_true & any_missing, pd.NA)
    return pd.Series(combined, index=index, dtype="boolean")


def _flag(values: np.ndarray, direction: str, lower: float | None, upper: float | None) -> pd.array:
    output = np.full(values.shape, pd.NA, dtype=object)
    present = np.isfinite(values)
    if direction == "lower":
        output[present] = values[present] < lower
    elif direction == "upper":
        output[present] = values[present] > upper
    else:
        output[present] = (values[present] < lower) | (values[present] > upper)
    return pd.array(output, dtype="boolean")


def _calculate_metric(
    raw: np.ndarray,
    *,
    metric: str,
    direction: str,
    n_mads: float,
    transform: str,
    min_n: int,
    zero_mad: str,
    index: pd.Index,
) -> _MetricResult:
    values = transform_values(raw, transform, metric)
    usable = values[np.isfinite(values)]
    status = "ok"
    centre = spread = lower = upper = None
    if not len(usable):
        status = "all_missing"
    elif len(usable) < min_n:
        status = "insufficient_n"
    else:
        centre, spread = mad(usable)
        if spread == 0:
            status = "zero_mad"
            if zero_mad == "error":
                raise ValueError(f"MAD is zero for metric `{metric}`.")
            if zero_mad == "zero":
                lower = centre if direction in {"lower", "both"} else None
                upper = centre if direction in {"upper", "both"} else None
        else:
            lower = centre - n_mads * spread if direction in {"lower", "both"} else None
            upper = centre + n_mads * spread if direction in {"upper", "both"} else None
    flag = pd.Series(
        _flag(values, direction, lower, upper)
        if status == "ok" or (status == "zero_mad" and zero_mad == "zero")
        else pd.array([pd.NA] * len(values), dtype="boolean"),
        index=index,
        name=metric,
    )
    threshold = {
        "metric": metric,
        "direction": direction,
        "transform": transform,
        "median": centre,
        "mad": spread,
        "lower": lower,
        "upper": upper,
        "lower_raw": inverse_transform(lower, transform),
        "upper_raw": inverse_transform(upper, transform),
        "status": status,
    }
    return _MetricResult(flag=flag, threshold=threshold, status=status)


def _build_metadata(
    threshold_rows: list[dict[str, object]],
    metrics: dict[str, str],
    transforms: dict[str, str],
    n_mads: float,
    min_n: int,
    zero_mad: str,
) -> dict[str, object]:
    numeric_fields = {"median", "mad", "lower", "upper", "lower_raw", "upper_raw"}
    threshold_metadata = {
        field: (
            np.array([row[field] if row[field] is not None else np.nan for row in threshold_rows])
            if field in numeric_fields
            else [row[field] for row in threshold_rows]
        )
        for field in threshold_rows[0]
    }
    return {
        "thresholds": threshold_metadata,
        "params": {
            "metrics": metrics,
            "transform": transforms,
            "n_mads": n_mads,
            "constant": CONSTANT,
            "min_n": min_n,
            "zero_mad": zero_mad,
        },
    }


def flag_mad_outliers(
    data: ad.AnnData | pd.DataFrame,
    *,
    metrics: Mapping[str, str],
    n_mads: float = 3.0,
    transform: str | Mapping[str, str] = "none",
    min_n: int = 5,
    zero_mad: str = "na",
    overwrite: bool = False,
    copy: bool = False,
) -> ad.AnnData | pd.DataFrame | None:
    """Flag explicit lower, upper, or both-tail MAD outliers.

    AnnData metrics are read from ``.obs``. No observations are filtered.
    AnnData is modified in place unless ``copy=True``; DataFrame input always
    returns an annotated copy.
    """
    if zero_mad not in {"na", "zero", "error"}:
        raise ValueError("`zero_mad` must be `na`, `zero`, or `error`.")
    n_mads = validate_positive(n_mads, "n_mads")
    min_n = validate_min_n(min_n)
    if not isinstance(overwrite, (bool, np.bool_)) or not isinstance(copy, (bool, np.bool_)):
        raise TypeError("`overwrite` and `copy` must be boolean.")

    is_adata = isinstance(data, ad.AnnData)
    if is_adata:
        target = data.copy() if copy else data
        table = target.obs
    elif isinstance(data, pd.DataFrame):
        target = data.copy()
        table = target
    else:
        raise TypeError("`data` must be an AnnData object or pandas DataFrame.")

    metrics = validate_metrics(metrics)
    missing = [metric for metric in metrics if metric not in table.columns]
    if missing:
        raise KeyError(f"Missing metric column(s): {', '.join(missing)}.")
    transforms = resolve_transforms(transform, list(metrics))
    targets = [*(f"{metric}_mad_outlier" for metric in metrics), "mad_outlier"]
    conflicts = [column for column in targets if column in table.columns]
    if conflicts and not overwrite:
        raise ValueError(
            f"Flag column(s) already exist: {', '.join(conflicts)}. Use `overwrite=True`."
        )

    flags: list[pd.Series] = []
    threshold_rows: list[dict[str, object]] = []
    insufficient: list[str] = []
    for metric, direction in metrics.items():
        raw = numeric_values(table[metric], metric)
        result = _calculate_metric(
            raw,
            metric=metric,
            direction=direction,
            n_mads=n_mads,
            transform=transforms[metric],
            min_n=min_n,
            zero_mad=zero_mad,
            index=table.index,
        )
        if result.status == "insufficient_n":
            insufficient.append(metric)
        flags.append(result.flag)
        threshold_rows.append(result.threshold)
    if insufficient:
        warnings.warn(
            f"Insufficient usable observations for metric(s): {', '.join(insufficient)}.",
            UserWarning,
            stacklevel=2,
        )

    combined = _combine_flags(flags, table.index)
    for metric, flag in zip(metrics, flags):
        table[f"{metric}_mad_outlier"] = flag
    table["mad_outlier"] = combined

    metadata = _build_metadata(threshold_rows, metrics, transforms, n_mads, min_n, zero_mad)
    if is_adata:
        target.uns["verymad"] = metadata
        return target if copy else None
    target.attrs["verymad"] = metadata
    return target
