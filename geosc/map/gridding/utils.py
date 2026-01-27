import numpy as np
from skgstat import Variogram


def idw_interpolate(x, y, values, grid_x, grid_y, power=2):
    """
    Inverse Distance Weighting (IDW) interpolation
    """
    x = np.asarray(x)
    y = np.asarray(y)
    v = np.asarray(values)

    gx = np.asarray(grid_x).ravel()
    gy = np.asarray(grid_y).ravel()

    out = np.zeros_like(gx, dtype=float)

    for i, (xx, yy) in enumerate(zip(gx, gy)):
        dx = x - xx
        dy = y - yy
        dist = np.sqrt(dx * dx + dy * dy)
        dist[dist == 0.0] = 1e-12

        w = 1.0 / (dist ** power)
        w /= w.sum()
        out[i] = np.sum(w * v)

    return out



def auto_variogram(x, y, values, model="spherical"):
    coords = np.vstack([x, y]).T

    V = Variogram(
        coords,
        values,
        model=model,
        n_lags=12,
        maxlag="median",
    )

    return {
        "range": V.parameters[0],
        "sill": V.parameters[1],
        "nugget": V.parameters[2],
    }