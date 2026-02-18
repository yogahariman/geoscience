"""Pore pressure regression with target transform from effective pressure ratio."""

from __future__ import annotations

from typing import Any, Dict, Optional
import pickle

import numpy as np

from .regression import Regressor


class PorePressureRegressor(Regressor):
    """
    Specialized regressor for pore pressure prediction.

    Training target transform:
        eff_pressure = overburden - pore_pressure
        target = eff_pressure / (overburden - hydrostatic)

    Prediction inverse transform:
        eff_pressure = target * (overburden - hydrostatic)
        pore_pressure = overburden - eff_pressure
    """

    def __init__(
        self,
        model_type: str = "mlp",
        overburden_col: int = 0,
        hydrostatic_col: int = 1,
        use_scaler: Optional[bool] = None,
        model: Optional[Any] = None,
        scaler: Optional[Any] = None,
        y_scaler: Optional[Any] = None,
        scale_y: bool = False,
    ) -> None:
        super().__init__(
            model_type=model_type,
            use_scaler=use_scaler,
            model=model,
            scaler=scaler,
            y_scaler=y_scaler,
            scale_y=scale_y,
        )
        self.overburden_col = int(overburden_col)
        self.hydrostatic_col = int(hydrostatic_col)

    def _extract_overburden_hydrostatic(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features).")
        n_features = X.shape[1]
        if not (0 <= self.overburden_col < n_features):
            raise ValueError("overburden_col is out of feature range.")
        if not (0 <= self.hydrostatic_col < n_features):
            raise ValueError("hydrostatic_col is out of feature range.")
        return X[:, self.overburden_col], X[:, self.hydrostatic_col]

    @staticmethod
    def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        out = np.full(numerator.shape, np.nan, dtype=float)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
        out[valid] = numerator[valid] / denominator[valid]
        return out

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parameters: Optional[Dict[str, Any]] = None,
        scaler_params: Optional[Dict[str, Any]] = None,
        scale_x: Optional[bool] = None,
        scale_y: bool = False,
        y_scaler_params: Optional[Dict[str, Any]] = None,
    ) -> "PorePressureRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        overburden, hydrostatic = self._extract_overburden_hydrostatic(X)
        denom = overburden - hydrostatic
        eff_pressure = overburden - y
        target = self._safe_divide(eff_pressure, denom)

        invalid = np.isnan(target)
        if np.any(invalid):
            X = X[~invalid]
            target = target[~invalid]
        if X.shape[0] == 0:
            raise ValueError("No valid samples after target transform (check overburden/hydrostatic columns).")

        super().train(
            X=X,
            y=target,
            parameters=parameters,
            scaler_params=scaler_params,
            scale_x=scale_x,
            scale_y=scale_y,
            y_scaler_params=y_scaler_params,
        )
        return self

    def predict(
        self,
        X: np.ndarray,
        null_value: Optional[float] = None,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        overburden, hydrostatic = self._extract_overburden_hydrostatic(X)
        target_pred = super().predict(X, null_value=null_value)
        target_pred = np.asarray(target_pred, dtype=float)

        denom = overburden - hydrostatic
        eff_pressure = target_pred * denom
        pore_pressure = overburden - eff_pressure

        valid = (
            np.isfinite(overburden)
            & np.isfinite(hydrostatic)
            & np.isfinite(target_pred)
            & (np.abs(denom) > 1e-12)
        )
        if null_value is not None:
            null = float(null_value)
            valid &= (target_pred != null) & (overburden != null) & (hydrostatic != null)

        if null_value is None:
            out = np.full(X.shape[0], np.nan, dtype=float)
            out[valid] = pore_pressure[valid]
            return out

        out = np.full(X.shape[0], float(null_value), dtype=float)
        out[valid] = pore_pressure[valid]
        return out

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained. Nothing to save.")

        payload = {
            "model_type": self.model_type,
            "use_scaler": self.use_scaler,
            "model": self.model,
            "scaler": self.scaler,
            "y_scaler": self.y_scaler,
            "scale_y": self.scale_y,
            "overburden_col": self.overburden_col,
            "hydrostatic_col": self.hydrostatic_col,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "PorePressureRegressor":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        return cls(
            model_type=payload["model_type"],
            overburden_col=payload.get("overburden_col", 0),
            hydrostatic_col=payload.get("hydrostatic_col", 1),
            use_scaler=payload.get("use_scaler"),
            model=payload.get("model"),
            scaler=payload.get("scaler"),
            y_scaler=payload.get("y_scaler"),
            scale_y=payload.get("scale_y", False),
        )
