"""Shmin regression with target transform from effective stress ratio."""

from __future__ import annotations

from typing import Any, Dict, Optional
import pickle

import numpy as np

from .regression import Regressor


class ShminRegressor(Regressor):
    """
    Specialized regressor for Shmin prediction.

    Required first core inputs in X:
    - hydrostatic
    - overburden
    - porepressure

    Training transform:
        eff_shmin = shmin - porepressure
        target = eff_shmin / (overburden - porepressure)

    Prediction inverse transform:
        eff_shmin = target * (overburden - porepressure)
        shmin = porepressure + eff_shmin
    """

    def __init__(
        self,
        model_type: str = "mlp",
        hydrostatic_col: int = 0,
        overburden_col: int = 1,
        porepressure_col: int = 2,
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
        self.porepressure_col = int(porepressure_col)
        self.hydrostatic_col = int(hydrostatic_col)
        self.overburden_col = int(overburden_col)

    def _extract_core(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features).")
        n_features = X.shape[1]
        for idx_name, idx in [
            ("porepressure_col", self.porepressure_col),
            ("hydrostatic_col", self.hydrostatic_col),
            ("overburden_col", self.overburden_col),
        ]:
            if not (0 <= idx < n_features):
                raise ValueError(f"{idx_name} is out of feature range.")
        pp = X[:, self.porepressure_col]
        hydro = X[:, self.hydrostatic_col]
        ob = X[:, self.overburden_col]
        return pp, hydro, ob

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
    ) -> "ShminRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        pp, hydro, ob = self._extract_core(X)

        # Filter target exactly as required:
        # (shMin > porePressure) & (shMin < overburden)
        valid_target = (
            np.isfinite(y)
            & np.isfinite(pp)
            & np.isfinite(hydro)
            & np.isfinite(ob)
            & (y > pp)
            & (y < ob)
        )
        if np.any(~valid_target):
            X = X[valid_target]
            y = y[valid_target]
            pp = pp[valid_target]
            ob = ob[valid_target]
        if X.shape[0] == 0:
            raise ValueError("No valid samples after Shmin target filter.")

        eff_shmin = y - pp
        tt = self._safe_divide(eff_shmin, ob - pp)
        valid_tt = np.isfinite(tt)
        if np.any(~valid_tt):
            X = X[valid_tt]
            tt = tt[valid_tt]
        if X.shape[0] == 0:
            raise ValueError("No valid samples after Shmin target transform.")

        super().train(
            X=X,
            y=tt,
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
        pp, hydro, ob = self._extract_core(X)
        target_pred = super().predict(X, null_value=null_value)
        target_pred = np.asarray(target_pred, dtype=float)

        denom = ob - pp
        eff_shmin = target_pred * denom
        shmin = pp + eff_shmin

        valid = (
            np.isfinite(pp)
            & np.isfinite(hydro)
            & np.isfinite(ob)
            & np.isfinite(target_pred)
            & (np.abs(denom) > 1e-12)
        )
        if null_value is not None:
            null = float(null_value)
            valid &= (target_pred != null) & (pp != null) & (hydro != null) & (ob != null)

        if null_value is None:
            out = np.full(X.shape[0], np.nan, dtype=float)
            out[valid] = shmin[valid]
            return out

        out = np.full(X.shape[0], float(null_value), dtype=float)
        out[valid] = shmin[valid]
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
            "porepressure_col": self.porepressure_col,
            "hydrostatic_col": self.hydrostatic_col,
            "overburden_col": self.overburden_col,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "ShminRegressor":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        return cls(
            model_type=payload["model_type"],
            porepressure_col=payload.get("porepressure_col", 2),
            hydrostatic_col=payload.get("hydrostatic_col", 0),
            overburden_col=payload.get("overburden_col", 1),
            use_scaler=payload.get("use_scaler"),
            model=payload.get("model"),
            scaler=payload.get("scaler"),
            y_scaler=payload.get("y_scaler"),
            scale_y=payload.get("scale_y", False),
        )
