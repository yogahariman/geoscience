from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

from geosc.ml import DataCleaner


class NaiveBayesGaussianContour2D:
    """
    Generic 2D Gaussian contour plotter for Gaussian Naive Bayes.

    This class is domain-agnostic and can be used for well logs, XRD,
    laboratory data, or any 2-feature classification dataset.
    """

    def __init__(
        self,
        class_names: Optional[Dict[int, str]] = None,
        class_colors: Optional[Dict[int, str]] = None,
        null_value: Optional[float] = -999.25,
        var_smoothing: float = 1e-9,
    ) -> None:
        self.class_names = class_names or {}
        self.class_colors = class_colors or {}
        self.null_value = null_value
        self.var_smoothing = float(var_smoothing)

        self.model: Optional[GaussianNB] = None
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.feature_names_: Optional[Tuple[str, str]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        allowed_classes: Optional[Iterable[int]] = None,
    ) -> "NaiveBayesGaussianContour2D":
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must be shape (n_samples, 2).")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows.")

        if allowed_classes is not None:
            allowed = set(int(c) for c in allowed_classes)
            mask = np.array([int(v) in allowed for v in y], dtype=bool)
            X = X[mask]
            y = y[mask]

        cleaner = DataCleaner(null_value=self.null_value)
        X, y = cleaner.clean_data_training(X, y)
        if X.size == 0:
            raise ValueError("No valid rows available after filtering/cleaning.")

        model = GaussianNB(var_smoothing=self.var_smoothing)
        model.fit(X, y)

        self.model = model
        self.X_ = X
        self.y_ = y
        return self

    def fit_from_dataframe(
        self,
        df,
        feature_cols: Sequence[str],
        target_col: str,
        allowed_classes: Optional[Iterable[int]] = None,
    ) -> "NaiveBayesGaussianContour2D":
        if len(feature_cols) != 2:
            raise ValueError("feature_cols must contain exactly 2 columns.")

        X = df[list(feature_cols)].values
        y = df[target_col].values
        self.feature_names_ = (str(feature_cols[0]), str(feature_cols[1]))
        return self.fit(X=X, y=y, allowed_classes=allowed_classes)

    @staticmethod
    def _class_log_density(model: GaussianNB, class_index: int, points: np.ndarray) -> np.ndarray:
        mu = model.theta_[class_index]
        var = model.var_[class_index]
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * var) + ((points - mu) ** 2) / var,
            axis=1,
        )

    def plot(
        self,
        title: str = "Gaussian Contours Naive Bayes (2D)",
        figsize: Tuple[float, float] = (10, 8),
        grid_size: int = 300,
        contour_count: int = 6,
        percentiles: Tuple[float, float] = (70.0, 99.5),
        show_points: bool = True,
    ) -> None:
        if self.model is None or self.X_ is None or self.y_ is None:
            raise ValueError("Model is not fitted. Call fit() or fit_from_dataframe() first.")

        X = self.X_
        y = self.y_
        model = self.model

        pad_x = 0.05 * (X[:, 0].max() - X[:, 0].min() + 1e-12)
        pad_y = 0.05 * (X[:, 1].max() - X[:, 1].min() + 1e-12)
        x_min, x_max = X[:, 0].min() - pad_x, X[:, 0].max() + pad_x
        y_min, y_max = X[:, 1].min() - pad_y, X[:, 1].max() + pad_y

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_size),
            np.linspace(y_min, y_max, grid_size),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        plt.figure(figsize=figsize)

        classes = model.classes_
        for ci, c in enumerate(classes):
            c_int = int(c)
            cname = self.class_names.get(c_int, f"Class {c_int}")
            color = self.class_colors.get(c_int, None)

            if show_points:
                mask = y == c
                plt.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    s=18,
                    alpha=0.65,
                    c=color,
                    label=f"{cname} ({c_int})",
                    edgecolors="none",
                )

            logp = self._class_log_density(model, ci, grid)
            z = np.exp(logp).reshape(xx.shape)
            positive = z[z > 0]
            if positive.size == 0:
                continue

            zmin = np.percentile(positive, percentiles[0])
            zmax = np.percentile(positive, percentiles[1])
            if zmin <= 0 or zmax <= zmin:
                continue

            levels = np.geomspace(zmin, zmax, contour_count)
            plt.contour(
                xx,
                yy,
                z,
                levels=levels,
                colors=[color or "gray"],
                linewidths=1.5,
                alpha=0.9,
            )

        xname, yname = self.feature_names_ if self.feature_names_ is not None else ("X1", "X2")
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.title(title)
        if show_points:
            plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

