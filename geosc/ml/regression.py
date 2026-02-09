"""Regression utilities for continuous targets."""

from __future__ import annotations

from typing import Any, Dict, Optional
import pickle

import numpy as np

SUPPORTED_MODEL_TYPES = {
    "xgboost",
    "random_forest",
    "mlp",
    "svm",
}


def _require_sklearn():
    try:
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
        from sklearn.neural_network import MLPRegressor  # type: ignore
        from sklearn.svm import SVR  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Regressor requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    return StandardScaler, RandomForestRegressor, MLPRegressor, SVR


class Regressor:
    """
    Train and run regression models for continuous targets.

    Parameters
    ----------
    model_type : str
        One of: xgboost, random_forest, mlp, svm.
    use_scaler : bool, optional
        If None, scaler is enabled for mlp and svm.
    """

    def __init__(
        self,
        model_type: str = "mlp",
        use_scaler: Optional[bool] = None,
        model: Optional[Any] = None,
        scaler: Optional[Any] = None,
        y_scaler: Optional[Any] = None,
        scale_y: bool = False,
    ) -> None:
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type={model_type!r}. "
                f"Use one of {sorted(SUPPORTED_MODEL_TYPES)}"
            )

        self.model_type = model_type
        self.use_scaler = use_scaler if use_scaler is not None else model_type in {"mlp", "svm"}
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.scale_y = scale_y

    def _build_model(self, parameters: Dict[str, Any]) -> Any:
        StandardScaler, RandomForestRegressor, MLPRegressor, SVR = _require_sklearn()

        if self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor  # type: ignore
            except Exception as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "XGBoost model_type requires xgboost. Install with: pip install xgboost"
                ) from exc
            return XGBRegressor(**parameters)
        if self.model_type == "random_forest":
            return RandomForestRegressor(**parameters)
        if self.model_type == "mlp":
            return MLPRegressor(**parameters)
        if self.model_type == "svm":
            return SVR(**parameters)

        raise ValueError(f"Unsupported model_type={self.model_type!r}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parameters: Optional[Dict[str, Any]] = None,
        scaler_params: Optional[Dict[str, Any]] = None,
        scale_x: Optional[bool] = None,
        scale_y: bool = False,
        y_scaler_params: Optional[Dict[str, Any]] = None,
    ) -> "Regressor":
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features)")

        use_scaler = self.use_scaler if scale_x is None else scale_x
        if use_scaler:
            StandardScaler, *_ = _require_sklearn()
            self.scaler = StandardScaler(**(scaler_params or {}))
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = None

        self.scale_y = scale_y
        if scale_y:
            StandardScaler, *_ = _require_sklearn()
            self.y_scaler = StandardScaler(**(y_scaler_params or {}))
            y = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            self.y_scaler = None

        self.model = self._build_model(parameters or {})
        self.model.fit(X, y)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def _inverse_y(self, y: np.ndarray) -> np.ndarray:
        if self.scale_y and self.y_scaler is not None:
            return self.y_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
        return y

    def predict(
        self,
        X: np.ndarray,
        null_value: Optional[float] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained. Call train() or load() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features)")

        invalid_mask = np.any(np.isnan(X), axis=1)
        if invalid_mask.any():
            valid_mask = ~invalid_mask
            predictions = np.full(X.shape[0], np.nan, dtype=float)

            if valid_mask.any():
                X_t = self._transform(X[valid_mask])
                pred_valid = self.model.predict(X_t)
                pred_valid = self._inverse_y(np.asarray(pred_valid, dtype=float))
                predictions[valid_mask] = pred_valid

            if null_value is not None:
                predictions = np.where(invalid_mask, null_value, predictions)

            return predictions

        X_t = self._transform(X)
        predictions = self.model.predict(X_t)
        predictions = self._inverse_y(np.asarray(predictions, dtype=float))
        return predictions

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
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Regressor":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            model_type=payload["model_type"],
            use_scaler=payload.get("use_scaler"),
            model=payload.get("model"),
            scaler=payload.get("scaler"),
            y_scaler=payload.get("y_scaler"),
            scale_y=payload.get("scale_y", False),
        )
        return obj
