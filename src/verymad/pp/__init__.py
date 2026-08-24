"""Preprocessing helpers."""

from ._mad_outliers import flag_mad_outliers
from ._scale import mad_scale

__all__ = ["flag_mad_outliers", "mad_scale"]
