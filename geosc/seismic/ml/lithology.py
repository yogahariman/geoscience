from __future__ import annotations

from .classification import SeismicClassifier


class SeismicLithologyPredictor(SeismicClassifier):
    """
    Backward-compatible alias for lithology-focused naming.

    This class keeps the historical API while inheriting full behavior from
    ``SeismicClassifier``.
    """

    pass
