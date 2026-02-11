import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba


def build_discrete_style(values, class_colors):
    classes = np.asarray(values, dtype=float)
    if classes.size == 0:
        raise ValueError("Tidak ada class valid untuk diplot.")

    classes = np.sort(np.unique(classes))

    fallback = list(plt.get_cmap("tab20").colors)
    colors = []
    for i, c in enumerate(classes):
        key_int = int(round(float(c)))
        if key_int in class_colors:
            colors.append(to_rgba(class_colors[key_int]))
        elif float(c) in class_colors:
            colors.append(to_rgba(class_colors[float(c)]))
        else:
            colors.append(fallback[i % len(fallback)])

    if classes.size == 1:
        bounds = np.array([classes[0] - 0.5, classes[0] + 0.5], dtype=float)
    else:
        mids = (classes[:-1] + classes[1:]) / 2.0
        first = classes[0] - (mids[0] - classes[0])
        last = classes[-1] + (classes[-1] - mids[-1])
        bounds = np.concatenate([[first], mids, [last]])

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, classes


def mask_null_values(data, null_value):
    out = np.asarray(data, dtype=float).copy()
    out[out == null_value] = np.nan
    return np.ma.masked_invalid(out)

