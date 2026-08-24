<h1 align="center">veryMADpy</h1>

<p align="center">
  <strong>Explicit MAD quality control for Python and AnnData</strong>
</p>

<p align="center">
  <a href="https://github.com/Thokas99/veryMADpy/actions/workflows/tests.yml"><img src="https://github.com/Thokas99/veryMADpy/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/Thokas99/veryMADpy/releases/latest"><img src="https://img.shields.io/github/v/release/Thokas99/veryMADpy?display_name=tag&sort=semver" alt="GitHub Release"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-%3E%3D3.10-blue.svg" alt="Python >=3.10"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
</p>

veryMADpy is a small Python/scverse-native toolkit for explicit median absolute deviation (MAD) quality-control flagging and robust scaling. It is the Python sibling of [veryMAD](https://github.com/Thokas99/veryMAD) and integrates directly with AnnData and Scanpy-style QC metadata.

veryMADpy calculates transparent statistical thresholds and flags observations. It never automatically removes observations or turns a statistical flag into a biological diagnosis.

## Capabilities

| Capability | Function | Input |
| --- | --- | --- |
| Observation-level MAD QC | `vm.pp.flag_mad_outliers()` | `AnnData`, `DataFrame` |
| Robust MAD scaling | `vm.pp.mad_scale()` | NumPy, Series, DataFrame |

For AnnData, QC metrics are read from `adata.obs`. Flags are written to `adata.obs` and threshold metadata is stored in `adata.uns["verymad"]`.

## Installation

veryMADpy is not on PyPI yet. Install the development version directly from GitHub:

```bash
python -m pip install git+https://github.com/Thokas99/veryMADpy.git
```

For the reproducible `v0.0.1` release:

```bash
python -m pip install "git+https://github.com/Thokas99/veryMADpy.git@v0.0.1"
```

The repository/project name is `veryMADpy`; the Python import namespace is `verymad`:

```python
import verymad as vm
```

## Quick start: Scanpy and AnnData

Scanpy calculates QC metrics. veryMADpy calculates robust adaptive thresholds and flags the observations selected by you.

```python
import scanpy as sc
import verymad as vm

adata.var["mt"] = adata.var_names.str.startswith("MT-")

sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt"],
    percent_top=None,
    inplace=True,
)

vm.pp.flag_mad_outliers(
    adata,
    metrics={
        "total_counts": "lower",
        "n_genes_by_counts": "lower",
        "pct_counts_mt": "upper",
    },
    transform={
        "total_counts": "log1p",
        "n_genes_by_counts": "log1p",
    },
)
```

Inspect the per-observation flags and calculation metadata:

```python
adata.obs[
    [
        "total_counts_mad_outlier",
        "n_genes_by_counts_mad_outlier",
        "pct_counts_mt_mad_outlier",
        "mad_outlier",
    ]
]

adata.uns["verymad"]
```

`adata.obs` contains per-observation nullable flags. Each flag can be `True`, `False`, or missing when classification is undefined. `adata.uns["verymad"]` contains the thresholds and calculation parameters. veryMADpy complements `sc.pp.calculate_qc_metrics()`; it does not replace it.

## DataFrame workflow

veryMADpy also works with observation-level omics metadata outside single-cell workflows:

```python
annotated = vm.pp.flag_mad_outliers(
    metadata,
    metrics={
        "library_size": "lower",
        "mapping_rate": "lower",
    },
    transform={
        "library_size": "log1p",
    },
)
```

## Robust scaling

```python
scaled = vm.pp.mad_scale(x, axis=0)
```

`axis=0` scales columns and `axis=1` scales rows. `mad_scale()` accepts NumPy arrays, pandas Series, and pandas DataFrames; pass a matrix or layer explicitly when working with AnnData.

## Design principles

- Metrics are selected explicitly.
- Tail directions are explicit: `lower`, `upper`, or `both`.
- Supported transforms are `none`, `log1p`, and `log10`.
- Missing and undefined classifications remain missing.
- Observations are never filtered automatically.
- veryMADpy does not calculate Scanpy QC metrics.
- Adaptive MAD thresholds are statistical heuristics, not biological decisions.

## Relationship to veryMAD

[veryMAD](https://github.com/Thokas99/veryMAD) is the R implementation and statistical reference. veryMADpy is its Python/scverse-native sibling. The statistical contract is intentionally aligned, with naming adapted to Python and Scanpy conventions rather than copied literally:

| R | Python |
| --- | --- |
| `mad_qc()` | `vm.pp.flag_mad_outliers()` |
| `margin` | `axis` |
| Seurat integration | AnnData integration |

Live R/Python parity tests run when R and the sibling reference package are available.

## Development

```bash
git clone https://github.com/Thokas99/veryMADpy.git
cd veryMADpy

python -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev,test-scanpy]"

pytest -q
ruff check .
ruff format --check .
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## License

MIT. See [LICENSE](LICENSE).
