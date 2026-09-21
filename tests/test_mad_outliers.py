import numpy as np
import pandas as pd
import pytest

from verymad.pp import flag_mad_outliers
from verymad.pp._mad_outliers import _calculate_metric


def test_tails_and_thresholds_are_explicit():
    data = pd.DataFrame({"low": [1, 10, 11, 12, 13], "high": [1, 2, 3, 4, 20]})
    result = flag_mad_outliers(data, metrics={"low": "lower", "high": "upper"}, n_mads=1)
    assert result is not data
    assert result["low_mad_outlier"].dtype == "boolean"
    assert result.loc[0, "low_mad_outlier"]
    assert result.loc[4, "high_mad_outlier"]
    assert result["mad_outlier"].dtype == "boolean"
    assert list(data.columns) == ["low", "high"]


def test_transform_partial_override_and_raw_thresholds():
    data = pd.DataFrame({"counts": [1, 2, 3, 4, 100], "genes": [1, 2, 3, 4, 5]})
    result = flag_mad_outliers(
        data,
        metrics={"counts": "upper", "genes": "lower"},
        transform={"counts": "log1p"},
        n_mads=1,
    )
    assert result.loc[4, "counts_mad_outlier"]
    median = np.log1p(3)
    spread = 1.4826 * np.median(np.abs(np.log1p([1, 2, 3, 4, 100]) - median))
    assert np.expm1(median + spread) > 4


def test_log10_and_both_directions():
    data = pd.DataFrame({"x": [1, 10, 100, 1000, 100000]})
    result = flag_mad_outliers(data, metrics={"x": "both"}, transform="log10", n_mads=1)
    assert bool(result.loc[0, "x_mad_outlier"])
    assert bool(result.loc[4, "x_mad_outlier"])


def test_three_state_combination():
    data = pd.DataFrame({"good": [1, 2, 3, 4, 100], "missing": [np.nan] * 5})
    result = flag_mad_outliers(data, metrics={"good": "upper", "missing": "upper"}, min_n=3)
    assert bool(result.loc[4, "mad_outlier"])
    assert pd.isna(result.loc[0, "mad_outlier"])


def test_invalid_metric_and_transform_inputs():
    data = pd.DataFrame({"x": range(5), "label": list("abcde")})
    with pytest.raises(KeyError, match="Missing"):
        flag_mad_outliers(data, metrics={"nope": "lower"})
    with pytest.raises(TypeError, match="numeric"):
        flag_mad_outliers(data, metrics={"label": "lower"})
    with pytest.raises(ValueError, match="non-negative"):
        flag_mad_outliers(
            pd.DataFrame({"x": [-1, 0, 1, 2, 3]}), metrics={"x": "lower"}, transform="log1p"
        )
    with pytest.raises(ValueError, match="positive"):
        flag_mad_outliers(
            pd.DataFrame({"x": [0, 1, 2, 3, 4]}), metrics={"x": "lower"}, transform="log10"
        )
    with pytest.raises(ValueError, match="non-finite"):
        flag_mad_outliers(pd.DataFrame({"x": [1, 2, 3, 4, np.inf]}), metrics={"x": "lower"})


def test_metric_calculation_returns_transformed_and_raw_thresholds():
    result = _calculate_metric(
        np.array([1.0, 2.0, 3.0, 4.0, 100.0]),
        metric="counts",
        direction="upper",
        n_mads=1.0,
        transform="log1p",
        min_n=5,
        zero_mad="na",
        index=pd.Index(["a", "b", "c", "d", "e"]),
    )

    assert result.status == "ok"
    assert result.flag.dtype == "boolean"
    assert bool(result.flag.loc["e"])
    assert result.threshold["transform"] == "log1p"
    assert result.threshold["upper_raw"] > result.threshold["upper"]
