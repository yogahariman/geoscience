import numpy as np
from .base import BaseGridder
from .utils import idw_interpolate, auto_variogram

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging


class KrigingGridder(BaseGridder):

    def fit(self, x, y, values):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.v = np.asarray(values)

        self.model = self.params.get("variogram_model", "spherical")
        manual = self.params.get("variogram_parameters", None)

        # auto variogram jika tidak di-set
        if manual is None:
            self.variogram_parameters = auto_variogram(self.x, self.y, self.v, self.model)
        else:
            self.variogram_parameters = manual

        self.uk = self.params.get("universal", False)

    def predict(self, x, y, values, grid_x, grid_y):
        self.fit(x, y, values)

        # Shape asli (supaya nanti dikembalikan)
        orig_shape = grid_x.shape

        # Flatten semua titik
        gx = np.asarray(grid_x).flatten()
        gy = np.asarray(grid_y).flatten()

        try:
            if self.uk:
                K = UniversalKriging(
                    self.x, self.y, self.v,
                    variogram_model=self.model,
                    variogram_parameters=self.variogram_parameters,
                    drift_terms=self.params.get("drift_terms", ["regional_linear"])
                )
            else:
                K = OrdinaryKriging(
                    self.x, self.y, self.v,
                    variogram_model=self.model,
                    variogram_parameters=self.variogram_parameters,
                )

            # EKSEKUSI UNTUK ARBITRARY POINTS
            z, _ = K.execute("points", gx, gy)

        except Exception:
            # fallback
            z = idw_interpolate(x, y, values, grid_x, grid_y)

        # reshape balik
        return z.reshape(orig_shape)
