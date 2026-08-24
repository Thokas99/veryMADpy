import shutil
import subprocess
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from verymad.pp import flag_mad_outliers, mad_scale

ROOT = Path(__file__).parents[1]
REFERENCE = ROOT.parent / "veryMAD"
R_AVAILABLE = shutil.which("Rscript") is not None and (REFERENCE / "R" / "mad-qc.R").exists()


def _run_r(code: str) -> list[str]:
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _r_qc_cases() -> dict[str, list[str]]:
    reference = str(REFERENCE).replace("\\", "/")
    code = f'''
source("{reference}/R/mad-qc.R")
source("{reference}/R/mad-scale.R")
fmt <- function(x) if (length(x) == 0 || is.na(x)) "NA" else sprintf("%.17g", as.numeric(x))
flag_fmt <- function(x) if (is.na(x)) "NA" else if (isTRUE(x)) "TRUE" else "FALSE"
emit <- function(name, x, direction, transformation="none", min_n=5, policy="na") {{
  result <- tryCatch(
    mad_qc(data.frame(x=x), c(x=direction), transform=transformation,
           min_n=min_n, zero_mad=policy, output="report", verbose=FALSE),
    error=function(e) structure(list(message=conditionMessage(e)), class="vm_error")
  )
  if (inherits(result, "vm_error")) {{
    cat(paste(name, "ERROR", sep="\\t"), "\\n", sep="")
    return(invisible(NULL))
  }}
  threshold <- result$thresholds[1, ]
  fields <- vapply(c("median", "mad", "lower", "upper", "lower_raw", "upper_raw"),
                   function(k) fmt(threshold[[k]][[1]]), character(1))
  flags <- vapply(result$flags$x, flag_fmt, character(1))
  cat(paste(c(name, "OK", fields, threshold$status, paste(flags, collapse=",")), collapse="\\t"), "\\n", sep="")
}}
base <- c(1, 2, 3, 4, 20)
for (direction in c("lower", "upper", "both")) for (transformation in c("none", "log1p", "log10"))
  emit(paste(transformation, direction, sep="_"), base, direction, transformation, min_n=5)
emit("missing", c(1, 2, NA_real_, 4, 20), "both", min_n=4)
emit("all_missing", rep(NA_real_, 5), "upper")
emit("insufficient", c(1, 2, NA_real_, NA_real_, NA_real_), "upper")
emit("zero_na", c(1, 2, 2, 2, 3), "both", policy="na")
emit("zero_zero", c(1, 2, 2, 2, 3), "both", policy="zero")
emit("zero_error", c(1, 2, 2, 2, 3), "both", policy="error")
emit_scale <- function(name, x) {{
  values <- if (is.null(dim(x))) as.numeric(x) else as.numeric(t(x))
  cat(paste(name, paste(vapply(values, fmt, character(1)), collapse=","), sep="\\t"), "\\n", sep="")
}}
emit_scale("scale_vector", mad_scale(c(1, 2, 100)))
emit_scale("scale_columns", mad_scale(matrix(1:12, nrow=3), margin=2))
emit_scale("scale_rows", mad_scale(matrix(1:12, nrow=3), margin=1))
emit_scale("scale_zero_na", mad_scale(c(1, 1, 1), zero_mad="na"))
emit_scale("scale_zero_zero", mad_scale(c(1, 1, 1), zero_mad="zero"))
'''
    return {line.split("\t", 1)[0]: line.split("\t")[1:] for line in _run_r(code)}


def _python_qc(values, direction, transform="none", min_n=5, zero_mad="na") -> list[str]:
    adata = ad.AnnData(
        np.ones((len(values), 1)),
        obs=pd.DataFrame({"x": values}, index=[f"cell{i}" for i in range(len(values))]),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            flag_mad_outliers(
                adata,
                metrics={"x": direction},
                transform=transform,
                min_n=min_n,
                zero_mad=zero_mad,
            )
    except ValueError:
        return ["ERROR"]
    threshold = adata.uns["verymad"]["thresholds"]
    fields = [
        "median",
        "mad",
        "lower",
        "upper",
        "lower_raw",
        "upper_raw",
    ]
    formatted = [
        "NA" if pd.isna(threshold[field][0]) else format(float(threshold[field][0]), ".17g")
        for field in fields
    ]
    flags = [
        "NA" if pd.isna(value) else "TRUE" if bool(value) else "FALSE"
        for value in adata.obs["x_mad_outlier"].array
    ]
    return ["OK", *formatted, threshold["status"][0], ",".join(flags)]


def _assert_r_python_qc(r_values, python_values):
    assert r_values[0] == python_values[0]
    if r_values[0] == "ERROR":
        return
    assert r_values[-2:] == python_values[-2:]
    for expected, observed in zip(r_values[1:-2], python_values[1:-2]):
        if expected == "NA":
            assert observed == "NA"
        else:
            assert observed != "NA"
            np.testing.assert_allclose(float(observed), float(expected), rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(
    not R_AVAILABLE, reason="Rscript and the sibling veryMAD reference are unavailable"
)
def test_live_r_python_qc_parity():
    r_cases = _r_qc_cases()
    base = [1.0, 2.0, 3.0, 4.0, 20.0]
    for direction in ("lower", "upper", "both"):
        for transform in ("none", "log1p", "log10"):
            name = f"{transform}_{direction}"
            _assert_r_python_qc(
                r_cases[name],
                _python_qc(base, direction, transform=transform),
            )
    cases = [
        ("missing", [1.0, 2.0, np.nan, 4.0, 20.0], "both", "none", 4, "na"),
        ("all_missing", [np.nan] * 5, "upper", "none", 5, "na"),
        ("insufficient", [1.0, 2.0, np.nan, np.nan, np.nan], "upper", "none", 5, "na"),
        ("zero_na", [1.0, 2.0, 2.0, 2.0, 3.0], "both", "none", 5, "na"),
        ("zero_zero", [1.0, 2.0, 2.0, 2.0, 3.0], "both", "none", 5, "zero"),
        ("zero_error", [1.0, 2.0, 2.0, 2.0, 3.0], "both", "none", 5, "error"),
    ]
    for name, values, direction, transform, min_n, zero_mad in cases:
        _assert_r_python_qc(
            r_cases[name],
            _python_qc(values, direction, transform, min_n, zero_mad),
        )


@pytest.mark.skipif(
    not R_AVAILABLE, reason="Rscript and the sibling veryMAD reference are unavailable"
)
def test_live_r_python_scale_parity():
    r_values = _r_qc_cases()
    expected = {
        "scale_vector": mad_scale(pd.Series([1.0, 2.0, 100.0], index=["a", "b", "c"])).to_numpy(),
        "scale_columns": mad_scale(np.arange(1, 13, dtype=float).reshape(3, 4), axis=0).ravel(),
        "scale_rows": mad_scale(np.arange(1, 13, dtype=float).reshape(3, 4), axis=1).ravel(),
        "scale_zero_na": mad_scale(np.array([1.0, 1.0, 1.0]), zero_mad="na").ravel(),
        "scale_zero_zero": mad_scale(np.array([1.0, 1.0, 1.0]), zero_mad="zero").ravel(),
    }
    for name, values in expected.items():
        observed = [
            np.nan if value == "NA" else float(value) for value in r_values[name][0].split(",")
        ]
        np.testing.assert_allclose(observed, values, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_infinities_are_rejected_but_nan_is_missing():
    with pytest.raises(ValueError, match="non-finite"):
        flag_mad_outliers(pd.DataFrame({"x": [1, 2, 3, 4, np.inf]}), metrics={"x": "upper"})
    with pytest.raises(ValueError, match="non-finite"):
        flag_mad_outliers(pd.DataFrame({"x": [1, 2, 3, 4, -np.inf]}), metrics={"x": "upper"})
    result = flag_mad_outliers(
        pd.DataFrame({"x": [1, 2, 3, 4, np.nan]}), metrics={"x": "upper"}, min_n=4
    )
    assert pd.isna(result.loc[4, "x_mad_outlier"])
