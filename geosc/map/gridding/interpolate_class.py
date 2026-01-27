import numpy as np
from .interpolate_value import grid_value


# =========================================================
# PUBLIC API (CLASS)
# =========================================================

def grid_class(
    x, y, classes,
    grid_x, grid_y,
    method="kriging",
    class_list=None,
    **kwargs
):
    """
    Indicator kriging for categorical data (lithology / facies).

    Returns
    -------
    class_map : 1D ndarray
        Final class at grid points
    probmaps : dict
        Probability map per class
    """

    x = np.asarray(x)
    y = np.asarray(y)
    classes = np.asarray(classes)

    if class_list is None:
        class_list = np.unique(classes)

    probmaps = {}

    # indicator kriging per class
    for cls in class_list:
        indicator = (classes == cls).astype(float)

        prob = grid_value(
            x, y, indicator,
            grid_x, grid_y,
            method=method,
            **kwargs
        )

        probmaps[cls] = prob

    stack = np.stack(
        [probmaps[c] for c in class_list], axis=-1
    )

    idx = np.argmax(stack, axis=-1)
    class_map = np.asarray(class_list)[idx]

    return class_map, probmaps