import anndata as ad
import numpy as np
import pandas as pd
import pytest

from verymad.pp import flag_mad_outliers


def make_adata():
    obs = pd.DataFrame(
        {
            "total_counts": [1, 2, 3, 4, 100],
            "pct_counts_mt": [1, 2, 3, 4, 5],
            "missing": [np.nan] * 5,
        },
        index=[f"c{i}" for i in range(5)],
    )
    return ad.AnnData(np.ones((5, 2)), obs=obs)


def test_anndata_in_place_and_uns():
    adata = make_adata()
    result = flag_mad_outliers(
        adata, metrics={"total_counts": "upper", "pct_counts_mt": "lower"}, min_n=3
    )
    assert result is None
    assert {"total_counts_mad_outlier", "pct_counts_mt_mad_outlier", "mad_outlier"} <= set(
        adata.obs
    )
    assert list(adata.uns["verymad"]["thresholds"]["metric"]) == ["total_counts", "pct_counts_mt"]
    assert adata.uns["verymad"]["params"]["constant"] == 1.4826


def test_anndata_copy_does_not_modify_original(tmp_path):
    original = make_adata()
    copied = flag_mad_outliers(
        original, metrics={"total_counts": "upper", "missing": "upper"}, copy=True
    )
    assert copied is not original
    assert "mad_outlier" not in original.obs
    assert "mad_outlier" in copied.obs
    assert str(copied.obs["total_counts_mad_outlier"].dtype) == "boolean"
    assert str(copied.obs["mad_outlier"].dtype) == "boolean"
    assert copied.obs["missing_mad_outlier"].isna().all()
    path = tmp_path / "roundtrip.h5ad"
    copied.write_h5ad(path)
    loaded = ad.read_h5ad(path)
    assert "mad_outlier" in loaded.obs
    assert str(loaded.obs["total_counts_mad_outlier"].dtype) == "boolean"
    assert str(loaded.obs["mad_outlier"].dtype) == "boolean"
    assert loaded.obs["missing_mad_outlier"].isna().all()
    assert loaded.uns["verymad"]["params"]["zero_mad"] == "na"


def test_scanpy_qc_metrics_feed_verymad(tmp_path):
    sc = pytest.importorskip("scanpy", reason="Scanpy is optional and not installed")
    counts = np.array(
        [
            [10, 1, 5, 1, 2, 1],
            [8, 0, 4, 0, 2, 1],
            [12, 2, 3, 1, 1, 0],
            [9, 1, 2, 0, 4, 1],
            [15, 3, 6, 2, 1, 1],
            [11, 0, 7, 0, 2, 2],
            [7, 1, 1, 1, 3, 1],
            [50, 10, 10, 5, 3, 2],
        ],
        dtype=float,
    )
    adata = ad.AnnData(
        counts,
        var=pd.DataFrame(index=["G1", "MT-ND1", "G2", "MT-CO1", "G3", "G4"]),
    )
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    qc_columns = ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
    assert set(qc_columns) <= set(adata.obs.columns)
    before = adata.obs[qc_columns].copy(deep=True)

    flag_mad_outliers(
        adata,
        metrics={
            "total_counts": "lower",
            "n_genes_by_counts": "lower",
            "pct_counts_mt": "upper",
        },
        transform={"total_counts": "log1p", "n_genes_by_counts": "log1p"},
    )

    assert {
        "total_counts_mad_outlier",
        "n_genes_by_counts_mad_outlier",
        "pct_counts_mt_mad_outlier",
        "mad_outlier",
    } <= set(adata.obs.columns)
    pd.testing.assert_frame_equal(adata.obs[qc_columns], before)
    assert set(adata.uns["verymad"]) == {"thresholds", "params"}
    path = tmp_path / "scanpy-verymad.h5ad"
    adata.write_h5ad(path)
    loaded = ad.read_h5ad(path)
    assert "mad_outlier" in loaded.obs
    assert "verymad" in loaded.uns
