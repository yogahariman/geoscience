"""Clustering utilities for unsupervised learning."""

from __future__ import annotations

from typing import Any, Dict, Optional
import pickle

import numpy as np

SUPPORTED_MODEL_TYPES = {
    "kmeans",
    "dbscan",
    "gmm",
    "agglomerative",
    "som",
}


def _require_standard_scaler():
    try:
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Clusterer requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    return StandardScaler


def _require_sklearn_models():
    try:
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  # type: ignore
        from sklearn.mixture import GaussianMixture  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Clusterer requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    return KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture


class _SimpleSOM:
    """
    Minimal Self-Organizing Map (SOM) implementation.

    Produces cluster labels as flattened grid indices (row * n_cols + col).
    """

    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        n_iter: int = 1000,
        learning_rate: float = 0.5,
        sigma: float | None = None,
        random_state: int | None = None,
    ) -> None:
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive.")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.random_state = random_state

        self._weights: np.ndarray | None = None

    def _init_weights(self, X: np.ndarray) -> None:
        rng = np.random.RandomState(self.random_state)
        n_nodes = self.n_rows * self.n_cols
        if X.shape[0] >= n_nodes:
            indices = rng.choice(X.shape[0], size=n_nodes, replace=False)
            self._weights = X[indices].copy()
        else:
            self._weights = rng.normal(size=(n_nodes, X.shape[1]))

    def _bmu_index(self, x: np.ndarray) -> int:
        if self._weights is None:
            raise ValueError("SOM is not initialized.")
        diffs = self._weights - x
        dist = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dist))

    def fit(self, X: np.ndarray) -> "_SimpleSOM":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")

        self._init_weights(X)
        if self._weights is None:
            raise ValueError("Failed to initialize SOM weights.")

        n_nodes = self.n_rows * self.n_cols
        grid = np.array([(i // self.n_cols, i % self.n_cols) for i in range(n_nodes)])
        sigma0 = self.sigma if self.sigma is not None else max(self.n_rows, self.n_cols) / 2.0

        rng = np.random.RandomState(self.random_state)

        for it in range(self.n_iter):
            idx = rng.randint(0, X.shape[0])
            x = X[idx]
            bmu = self._bmu_index(x)

            lr = self.learning_rate * (1.0 - (it / max(1, self.n_iter - 1)))
            sigma_t = sigma0 * (1.0 - (it / max(1, self.n_iter - 1)))
            sigma_t = max(sigma_t, 1e-6)

            bmu_coord = grid[bmu]
            dists = np.sum((grid - bmu_coord) ** 2, axis=1)
            h = np.exp(-dists / (2.0 * (sigma_t ** 2)))

            self._weights += (lr * h[:, None]) * (x - self._weights)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if self._weights is None:
            raise ValueError("SOM is not fitted.")
        labels = np.empty(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            labels[i] = self._bmu_index(x)
        return labels


class Clusterer:
    """
    Train and run clustering models.

    Parameters
    ----------
    model_type : str
        One of: kmeans, dbscan, gmm, agglomerative.
    use_scaler : bool, optional
        If None, scaler is enabled.
    """

    def __init__(
        self,
        model_type: str = "kmeans",
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
        self.use_scaler = True if use_scaler is None else use_scaler
        self.model = model
        self.scaler = scaler

    def _build_model(self, parameters: Dict[str, Any]) -> Any:
        KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture = _require_sklearn_models()

        if self.model_type == "kmeans":
            return KMeans(**parameters)
        if self.model_type == "dbscan":
            return DBSCAN(**parameters)
        if self.model_type == "gmm":
            return GaussianMixture(**parameters)
        if self.model_type == "agglomerative":
            return AgglomerativeClustering(**parameters)
        if self.model_type == "som":
            return _SimpleSOM(**parameters)

        raise ValueError(f"Unsupported model_type={self.model_type!r}")

    def fit(
        self,
        X: np.ndarray,
        parameters: Optional[Dict[str, Any]] = None,
        scaler_params: Optional[Dict[str, Any]] = None,
        scale_x: Optional[bool] = None,
    ) -> "Clusterer":
        X = np.asarray(X)
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
        self.model.fit(X)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def fit_predict(
        self,
        X: np.ndarray,
        parameters: Optional[Dict[str, Any]] = None,
        scaler_params: Optional[Dict[str, Any]] = None,
        scale_x: Optional[bool] = None,
        null_value: Optional[float] = None,
    ) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features)")

        invalid_mask = np.any(np.isnan(X), axis=1)
        if invalid_mask.any():
            valid_mask = ~invalid_mask
            labels = np.full(X.shape[0], np.nan, dtype=float)

            if valid_mask.any():
                X_v = X[valid_mask]
                use_scaler = self.use_scaler if scale_x is None else scale_x
                if use_scaler:
                    StandardScaler = _require_standard_scaler()
                    self.scaler = StandardScaler(**(scaler_params or {}))
                    X_v = self.scaler.fit_transform(X_v)
                else:
                    self.scaler = None

                self.model = self._build_model(parameters or {})
                if hasattr(self.model, "fit_predict"):
                    labels_valid = self.model.fit_predict(X_v)
                else:
                    self.model.fit(X_v)
                    labels_valid = self.model.predict(X_v)

                labels[valid_mask] = labels_valid

            if null_value is not None:
                labels = np.where(invalid_mask, null_value, labels)

            return labels

        use_scaler = self.use_scaler if scale_x is None else scale_x
        if use_scaler:
            StandardScaler = _require_standard_scaler()
            self.scaler = StandardScaler(**(scaler_params or {}))
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = None

        self.model = self._build_model(parameters or {})
        if hasattr(self.model, "fit_predict"):
            return self.model.fit_predict(X)

        self.model.fit(X)
        return self.model.predict(X)

    def predict(
        self,
        X: np.ndarray,
        null_value: Optional[float] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() or fit_predict() first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features)")

        invalid_mask = np.any(np.isnan(X), axis=1)
        if invalid_mask.any():
            valid_mask = ~invalid_mask
            labels = np.full(X.shape[0], np.nan, dtype=float)

            if valid_mask.any():
                X_t = self._transform(X[valid_mask])
                if hasattr(self.model, "predict"):
                    labels_valid = self.model.predict(X_t)
                else:
                    raise ValueError("Model does not support predict().")
                labels[valid_mask] = labels_valid

            if null_value is not None:
                labels = np.where(invalid_mask, null_value, labels)

            return labels

        X_t = self._transform(X)
        if not hasattr(self.model, "predict"):
            raise ValueError("Model does not support predict().")
        return self.model.predict(X_t)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not fitted. Nothing to save.")

        payload = {
            "model_type": self.model_type,
            "use_scaler": self.use_scaler,
            "model": self.model,
            "scaler": self.scaler,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Clusterer":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        return cls(
            model_type=payload["model_type"],
            use_scaler=payload.get("use_scaler"),
            model=payload.get("model"),
            scaler=payload.get("scaler"),
        )
