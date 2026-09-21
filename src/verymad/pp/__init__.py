"""Preprocessing helpers."""

from ._mad_outliers import flag_mad_outliers
from ._scale import mad_scale
from ._summary import summarize_mad

__all__ = ["flag_mad_outliers", "mad_scale", "summarize_mad"]
