import anndata as ad
import numpy as np
import pandas as pd

from verymad.pp import flag_mad_outliers, summarize_mad


def test_summarize_mad_reports_dataframe_thresholds_and_counts():
    data = pd.DataFrame({"counts": [1, 2, 3, 4, 100], "missing": [np.nan] * 5})
    result = flag_mad_outliers(
        data,
        metrics={"counts": "upper", "missing": "upper"},
        transform={"counts": "log1p"},
        min_n=3,
    )

    summary = summarize_mad(result)

    assert list(summary["metric"]) == ["counts", "missing"]
    assert list(summary["status"]) == ["ok", "all_missing"]
    assert list(summary["flagged"]) == [1, 0]
    assert list(summary["undefined"]) == [0, 5]
    assert summary.loc[0, "upper_raw"] > summary.loc[0, "upper"]


def test_summarize_mad_supports_anndata_metadata():
    data = ad.AnnData(
        np.ones((5, 1)),
        obs=pd.DataFrame({"x": [1, 2, 3, 4, 100]}, index=[f"cell{i}" for i in range(5)]),
    )
    flag_mad_outliers(data, metrics={"x": "upper"}, n_mads=1)

    summary = summarize_mad(data)

    assert summary.loc[0, "metric"] == "x"
    assert summary.loc[0, "flagged"] == 1
    assert summary.loc[0, "undefined"] == 0
