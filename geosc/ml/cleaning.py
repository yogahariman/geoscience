"""Shared data cleaning utilities for training and prediction workflows."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class DataCleaner:
    """
    Clean X/y arrays across domains (well, seismic, map, etc.).

    Note: Training and prediction have different behaviors.
    """

    def __init__(self, null_value: Optional[float] = -999.25) -> None:
        self.null_value = null_value

    def clean_data_training(
        self,
        X,
        y,
        null_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Replace null markers with NaN and drop rows containing NaN
        in either X or y. Returns cleaned (X, y).
        """

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        null_value = self.null_value if null_value is None else null_value
        if null_value is not None:
            X = np.where(X == null_value, np.nan, X)
            y = np.where(y == null_value, np.nan, y)

        invalid = np.any(pd.isna(X), axis=1)
        if y.ndim == 1:
            invalid = invalid | pd.isna(y)
        else:
            invalid = invalid | np.any(pd.isna(y), axis=1)

        if invalid.all():
            raise ValueError("All rows are invalid after cleaning.")

        return X[~invalid], y[~invalid]

    def clean_data_prediction(
        self,
        X,
        null_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Replace null markers with NaN. Does not drop rows.
        """

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        null_value = self.null_value if null_value is None else null_value
        if null_value is not None:
            X[X == null_value] = np.nan

        return X

    def clean_xy(
        self,
        X,
        y,
        null_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias for clean_data_training().
        """

        return self.clean_data_training(X, y, null_value=null_value)
