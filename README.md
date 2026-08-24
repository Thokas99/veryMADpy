# veryMADpy

veryMADpy is the Python/scverse-native sibling of [veryMAD](https://github.com/Thokas99/veryMAD). It calculates transparent median absolute deviation (MAD) thresholds and flags unusual observations without automatically filtering them.

Scanpy users should generally calculate QC metrics first with `sc.pp.calculate_qc_metrics()`, then select the metric directions explicitly:

```python
import scanpy as sc
import verymad as vm

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
vm.pp.flag_mad_outliers(
    adata,
    metrics={
        "total_counts": "lower",
        "n_genes_by_counts": "lower",
        "pct_counts_mt": "upper",
    },
)
```

Flags are written to `adata.obs` as `<metric>_mad_outlier` and `mad_outlier`; threshold metadata is written to `adata.uns["verymad"]`. No observations are removed.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
ruff check .
```

DataFrames are also supported:

```python
annotated = vm.pp.flag_mad_outliers(
    metadata,
    metrics={"total_counts": "lower"},
    transform="log1p",
)
```

For robust scaling, use `axis=0` for columns or `axis=1` for rows:

```python
scaled = vm.pp.mad_scale(adata.X, axis=0)
```

Missing or undefined classifications remain missing. A MAD flag is an adaptive statistical heuristic, not an observation-rejection rule, laboratory diagnosis, or biological conclusion. veryMADpy does not detect metrics, filter observations, calculate Scanpy QC metrics, normalize data, or provide plotting and workflow APIs.

## License

MIT. See [LICENSE](LICENSE).
