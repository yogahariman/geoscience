import numpy as np
from skgstat import Variogram


def idw_interpolate(x, y, values, grid_x, grid_y, power=2):
    x = np.asarray(x)
    y = np.asarray(y)
    v = np.asarray(values)

    gx = grid_x.flatten()
    gy = grid_y.flatten()

    grid_values = np.zeros_like(gx)

    for i, (xx, yy) in enumerate(zip(gx, gy)):
        dx = x - xx
        dy = y - yy
        dist = np.sqrt(dx*dx + dy*dy)
        dist[dist == 0] = 1e-12

        w = 1.0 / (dist**power)
        w /= w.sum()
        grid_values[i] = np.sum(w * v)

    return grid_values.reshape(grid_x.shape)


def auto_variogram(x, y, values, model="spherical"):
    coords = np.vstack([x, y]).T

    V = Variogram(
        coords,
        values,
        model=model,
        n_lags=12,
        maxlag="median",
    )

    # hasil variogram berupa dict sill, range, nugget
    return {
        "sill": V.parameters[1],
        "range": V.parameters[0],
        "nugget": V.parameters[2],
    }
