# Changelog

## [0.0.1] - 2026-08-24

Initial release of the Python/scverse-native sibling of [veryMAD](https://github.com/Thokas99/veryMAD).

### Added

- `vm.pp.flag_mad_outliers()` for explicit observation-level MAD QC.
- AnnData and pandas DataFrame support.
- Explicit `lower`, `upper`, and `both` tails.
- `none`, `log1p`, and `log10` transformations.
- Nullable three-state outlier flags.
- Threshold and calculation metadata in `adata.uns["verymad"]`.
- `vm.pp.mad_scale()` for NumPy and pandas inputs.
- Scanpy integration testing and `.h5ad` serialization testing.
- Live R/Python parity testing against veryMAD when R is available.
- GitHub Actions CI across supported Python versions.

[0.0.1]: https://github.com/Thokas99/veryMADpy/releases/tag/v0.0.1
