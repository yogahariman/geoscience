"""Classification utilities for categorical targets."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import pickle

import numpy as np

SUPPORTED_MODEL_TYPES = {
    "xgboost",
    "random_forest",
    "mlp",
    "svm",
    "naive_bayes",
}


def _require_standard_scaler():
    try:
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Classifier requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    return StandardScaler


def _require_sklearn_models():
    try:
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        from sklearn.neural_network import MLPClassifier  # type: ignore
        from sklearn.svm import SVC  # type: ignore
        from sklearn.naive_bayes import GaussianNB  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Classifier requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    return RandomForestClassifier, MLPClassifier, SVC, GaussianNB


class Classifier:
    """
    Train and run classification models.

    Parameters
    ----------
    model_type : str
        One of: xgboost, random_forest, mlp, svm, naive_bayes.
    use_scaler : bool, optional
        If None, scaler is enabled for mlp and svm.
    """

    def __init__(
        self,
        model_type: str = "mlp",
        use_scaler: Optional[bool] = None,
        model: Optional[Any] = None,
        scaler: Optional[Any] = None,
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
        self.classes_ = None

    def _build_model(self, parameters: Dict[str, Any]) -> Any:
        RandomForestClassifier, MLPClassifier, SVC, GaussianNB = _require_sklearn_models()

        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier  # type: ignore
            except Exception as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "XGBoost model_type requires xgboost. Install with: pip install xgboost"
                ) from exc
            return XGBClassifier(**parameters)
        if self.model_type == "random_forest":
            return RandomForestClassifier(**parameters)
        if self.model_type == "mlp":
            return MLPClassifier(**parameters)
        if self.model_type == "svm":
            params = {"probability": True}
            params.update(parameters)
            return SVC(**params)
        if self.model_type == "naive_bayes":
            return GaussianNB(**parameters)

        raise ValueError(f"Unsupported model_type={self.model_type!r}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parameters: Optional[Dict[str, Any]] = None,
        scaler_params: Optional[Dict[str, Any]] = None,
        scale_x: Optional[bool] = None,
    ) -> "Classifier":
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features)")

        use_scaler = self.use_scaler if scale_x is None else scale_x
        if use_scaler:
            StandardScaler = _require_standard_scaler()
            self.scaler = StandardScaler(**(scaler_params or {}))
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = None

        self.model = self._build_model(parameters or {})
        self.model.fit(X, y)

        if hasattr(self.model, "classes_"):
            self.classes_ = list(self.model.classes_)
        else:
            self.classes_ = list(np.unique(y))

        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def _predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            raise ValueError("Model is not trained. Call train() or load() first.")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(X))
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)

        return None

    def predict(
        self,
        X: np.ndarray,
        null_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.model is None:
            raise ValueError("Model is not trained. Call train() or load() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features)")

        invalid_mask = np.any(np.isnan(X), axis=1)
        if invalid_mask.any():
            valid_mask = ~invalid_mask
            predictions = np.full(X.shape[0], np.nan, dtype=float)
            probabilities = None

            if valid_mask.any():
                X_t = self._transform(X[valid_mask])
                pred_valid = self.model.predict(X_t)
                prob_valid = self._predict_proba(X_t)

                predictions = predictions.astype(object)
                predictions[valid_mask] = pred_valid

                if prob_valid is not None:
                    probabilities = np.full(
                        (X.shape[0], prob_valid.shape[1]), np.nan, dtype=float
                    )
                    probabilities[valid_mask, :] = prob_valid

            if null_value is not None:
                predictions = np.where(invalid_mask, null_value, predictions)
                if probabilities is not None:
                    probabilities[invalid_mask, :] = null_value

            return predictions, probabilities

        X_t = self._transform(X)
        predictions = self.model.predict(X_t)
        probabilities = self._predict_proba(X_t)
        return predictions, probabilities

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained. Nothing to save.")

        payload = {
            "model_type": self.model_type,
            "use_scaler": self.use_scaler,
            "model": self.model,
            "scaler": self.scaler,
            "classes_": self.classes_,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Classifier":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            model_type=payload["model_type"],
            use_scaler=payload.get("use_scaler"),
            model=payload.get("model"),
            scaler=payload.get("scaler"),
        )
        obj.classes_ = payload.get("classes_")
        return obj
