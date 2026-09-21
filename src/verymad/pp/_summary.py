from collections.abc import Mapping

import anndata as ad
import pandas as pd


def _metadata_for_position(values: object, position: int) -> object:
    if isinstance(values, (list, tuple)):
        return values[position]
    return values[position]  # type: ignore[index]


def summarize_mad(data: ad.AnnData | pd.DataFrame) -> pd.DataFrame:
    """Return a readable summary of thresholds, statuses, and flag counts."""
    if isinstance(data, ad.AnnData):
        table = data.obs
        metadata = data.uns.get("verymad")
    elif isinstance(data, pd.DataFrame):
        table = data
        metadata = data.attrs.get("verymad")
    else:
        raise TypeError("`data` must be an AnnData object or pandas DataFrame.")
    if not isinstance(metadata, Mapping) or "thresholds" not in metadata:
        raise KeyError("No veryMAD metadata found. Run `flag_mad_outliers` first.")

    thresholds = metadata["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise TypeError("Invalid veryMAD threshold metadata.")
    metrics = thresholds.get("metric", [])
    rows = []
    for position, metric in enumerate(metrics):
        metric = str(metric)
        flag_column = f"{metric}_mad_outlier"
        if flag_column not in table:
            raise KeyError(f"Missing flag column `{flag_column}`.")
        flags = table[flag_column]
        rows.append(
            {
                "metric": metric,
                "direction": _metadata_for_position(thresholds["direction"], position),
                "transform": _metadata_for_position(thresholds["transform"], position),
                "status": _metadata_for_position(thresholds["status"], position),
                "median": _metadata_for_position(thresholds["median"], position),
                "mad": _metadata_for_position(thresholds["mad"], position),
                "lower": _metadata_for_position(thresholds["lower"], position),
                "upper": _metadata_for_position(thresholds["upper"], position),
                "lower_raw": _metadata_for_position(thresholds["lower_raw"], position),
                "upper_raw": _metadata_for_position(thresholds["upper_raw"], position),
                "flagged": int(flags.fillna(False).sum()),
                "undefined": int(flags.isna().sum()),
            }
        )
    return pd.DataFrame(rows)
