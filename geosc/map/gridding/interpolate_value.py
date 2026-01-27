import numpy as np
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from .base import BaseGridder
from .utils import auto_variogram


class KrigingGridder(BaseGridder):
    """Kriging interpolation (Ordinary / Universal)."""

    def fit(self, x, y, values):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.v = np.asarray(values)

        self.model = self.params.get("variogram_model", "spherical")
        self.uk = self.params.get("universal", False)

        manual = self.params.get("variogram_parameters", None)
        if manual is None:
            self.variogram_parameters = auto_variogram(
                self.x, self.y, self.v, self.model
            )
        else:
            self.variogram_parameters = manual

    def predict(self, x, y, values, grid_x, grid_y):
        self.fit(x, y, values)

        gx = np.asarray(grid_x).ravel()
        gy = np.asarray(grid_y).ravel()

        if self.uk:
            K = UniversalKriging(
                self.x, self.y, self.v,
                variogram_model=self.model,
                variogram_parameters=self.variogram_parameters,
                drift_terms=self.params.get(
                    "drift_terms", ["regional_linear"]
                ),
            )
        else:
            K = OrdinaryKriging(
                self.x, self.y, self.v,
                variogram_model=self.model,
                variogram_parameters=self.variogram_parameters,
            )

        z, _ = K.execute("points", gx, gy)
        return z


# =========================================================
# PUBLIC API (VALUE)
# =========================================================

def grid_value(
    x, y, values,
    grid_x, grid_y,
    method="kriging",
    **kwargs
):
    """
    Grid continuous values.

    Supported methods
    -----------------
    - kriging  : Ordinary / Universal Kriging (pykrige)
    - idw      : Inverse Distance Weighting
    - nearest  : Nearest neighbour

    Returns
    -------
    z : 1D ndarray
        Interpolated values at (grid_x, grid_y)
    """

    method = method.lower()

    # --------------------------------------------------
    # KRIGING
    # --------------------------------------------------
    if method == "kriging":
        gridder = KrigingGridder(**kwargs)
        return gridder.predict(x, y, values, grid_x, grid_y)

    # --------------------------------------------------
    # IDW
    # --------------------------------------------------
    elif method == "idw":
        from .utils import idw_interpolate
        power = kwargs.get("power", 2)
        return idw_interpolate(x, y, values, grid_x, grid_y, power=power)

    # --------------------------------------------------
    # NEAREST NEIGHBOUR
    # --------------------------------------------------
    elif method == "nearest":
        from scipy.spatial import cKDTree
        x = np.asarray(x)
        y = np.asarray(y)
        v = np.asarray(values)

        tree = cKDTree(np.c_[x, y])
        _, idx = tree.query(np.c_[grid_x, grid_y], k=1)
        return v[idx]

    else:
        raise ValueError(f"Unknown gridding method: {method}")