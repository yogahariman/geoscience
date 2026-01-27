import numpy as np
from .interpolate_value import grid_value


def grid_class(x, y, classes, grid_x, grid_y, method="kriging", class_list=None, **kwargs):
    if class_list is None:
        class_list = np.unique(classes)

    probmaps = {}

    for cls in class_list:
        mask = (classes == cls).astype(float)

        probmaps[cls] = grid_value(
            x, y, mask,
            grid_x, grid_y,
            method=method,
            **kwargs
        )

    stack = np.stack([probmaps[c] for c in class_list], axis=-1)
    idx = np.argmax(stack, axis=-1)
    class_map = np.array(class_list)[idx]

    return class_map, probmaps
