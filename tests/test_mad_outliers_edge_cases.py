import numpy as np
import pandas as pd
import pytest

from verymad.pp import flag_mad_outliers


def test_empty_all_missing_and_insufficient_statuses():
    empty = flag_mad_outliers(pd.DataFrame({"x": pd.Series(dtype=float)}), metrics={"x": "both"})
    assert len(empty) == 0
    assert empty["mad_outlier"].dtype == "boolean"
    missing = flag_mad_outliers(pd.DataFrame({"x": [np.nan] * 5}), metrics={"x": "both"})
    assert missing["x_mad_outlier"].isna().all()
    with pytest.warns(UserWarning, match="Insufficient"):
        insufficient = flag_mad_outliers(
            pd.DataFrame({"x": [1, 2, np.nan, np.nan, np.nan]}), metrics={"x": "both"}
        )
    assert insufficient["x_mad_outlier"].isna().all()


def test_zero_mad_policies():
    data = pd.DataFrame({"x": [1, 2, 2, 2, 3]})
    na = flag_mad_outliers(data, metrics={"x": "both"}, zero_mad="na")
    assert na["x_mad_outlier"].isna().all()
    zero = flag_mad_outliers(data, metrics={"x": "both"}, zero_mad="zero")
    assert bool(zero.loc[0, "x_mad_outlier"])
    assert not bool(zero.loc[2, "x_mad_outlier"])
    with pytest.raises(ValueError, match="x"):
        flag_mad_outliers(data, metrics={"x": "both"}, zero_mad="error")


def test_conflicts_are_reported_together_and_overwrite_is_explicit():
    data = pd.DataFrame(
        {
            "a": range(5),
            "b": range(5),
            "a_mad_outlier": False,
            "b_mad_outlier": False,
            "mad_outlier": False,
        }
    )
    with pytest.raises(ValueError, match="a_mad_outlier.*b_mad_outlier.*mad_outlier"):
        flag_mad_outliers(data, metrics={"a": "lower", "b": "upper"})
    result = flag_mad_outliers(data, metrics={"a": "lower", "b": "upper"}, overwrite=True)
    assert result["a_mad_outlier"].dtype == "boolean"


def test_index_is_preserved():
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5]}, index=["a", "b", "c", "d", "e"])
    result = flag_mad_outliers(data, metrics={"x": "lower"})
    assert result.index.tolist() == data.index.tolist()
