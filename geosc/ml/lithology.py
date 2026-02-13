"""Lithology prediction utilities."""

from __future__ import annotations

from .classification import Classifier


class LithologyPredictor(Classifier):
    """
    Backward-compatible alias for lithology-focused naming.

    This class keeps the historical API while inheriting full behavior from
    ``Classifier``.
    """

    pass
