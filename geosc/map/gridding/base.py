class BaseGridder:
    """Base class for grid interpolation methods."""

    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, x, y, values):
        pass

    def predict(self, x, y, values, grid_x, grid_y):
        raise NotImplementedError