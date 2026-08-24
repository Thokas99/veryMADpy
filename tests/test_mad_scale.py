import numpy as np
import pandas as pd
import pytest

from verymad.pp import mad_scale


def test_vector_series_and_dataframe_preserve_container_metadata():
    array = mad_scale(np.array([1.0, 2.0, 100.0]))
    assert isinstance(array, np.ndarray)
    series = pd.Series([1.0, 2.0, 100.0], index=["a", "b", "c"], name="x")
    scaled_series = mad_scale(series)
    assert isinstance(scaled_series, pd.Series)
    assert scaled_series.index.equals(series.index)
    assert scaled_series.name == "x"
    frame = pd.DataFrame({"a": [1.0, 2.0, 100.0], "b": [2.0, 4.0, 6.0]}, index=["r1", "r2", "r3"])
    scaled_frame = mad_scale(frame)
    assert isinstance(scaled_frame, pd.DataFrame)
    assert scaled_frame.index.equals(frame.index)
    assert scaled_frame.columns.equals(frame.columns)


def test_axis_and_center_scale_options():
    values = np.arange(1, 13, dtype=float).reshape(3, 4)
    rows = mad_scale(values, axis=1)
    columns = mad_scale(values, axis=0)
    assert rows.shape == values.shape
    assert columns.shape == values.shape
    np.testing.assert_allclose(mad_scale(values, center=False, scale=False), values)
    assert not np.allclose(mad_scale(values, axis=0), mad_scale(values, axis=1))
    np.testing.assert_allclose(
        mad_scale(values, constant=1), (values - np.median(values, axis=0)) / 4
    )


def test_missing_values_and_zero_mad_policies():
    values = pd.Series([1.0, 2.0, 3.0, np.nan])
    assert pd.isna(mad_scale(values, na_rm=False).iloc[3])
    assert mad_scale(pd.Series([1.0, 1.0, 1.0]), zero_mad="zero").tolist() == [0.0, 0.0, 0.0]
    assert mad_scale(pd.Series([1.0, 1.0, 1.0]), zero_mad="na").isna().all()
    with pytest.raises(ValueError, match="zero"):
        mad_scale(pd.Series([1.0, 1.0, 1.0]), zero_mad="error")


def test_scale_rejects_invalid_values():
    with pytest.raises(ValueError, match="Inf"):
        mad_scale(np.array([1.0, np.inf]))
    with pytest.raises(TypeError, match="numeric"):
        mad_scale(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    with pytest.raises(ValueError, match="axis"):
        mad_scale(np.ones((2, 2)), axis=2)
