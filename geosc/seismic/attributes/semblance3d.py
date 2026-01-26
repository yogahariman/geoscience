from .base.window import WindowAttribute
import numpy as np


class Semblance3D(WindowAttribute):
    """
    3D semblance attribute.
    Window = (wz, wx, wy)
    Geometry MUST be provided explicitly (inline, xline).
    """

    def __init__(
        self,
        input_segy,
        output_segy,
        window,
        inline,
        xline,
        load_to_ram=False,   # opsional preload RAM
    ):
        super().__init__(
            input_segy=input_segy,
            output_segy=output_segy,
            window=window,
            inline=inline,
            xline=xline,
            load_to_ram=load_to_ram,
        )

    def compute(self, D: np.ndarray) -> float:
        """
        D shape: (nz_window, ntraces_window)

        Semblance Taner (generalized for 3D):
            S = ( (sum amplitude all traces)^2 ) /
                ( N * sum of squares all traces )
        where:
            N = number of traces inside window
        """

        # Window terlalu kecil → semblance = 1
        if D.size == 0 or D.shape[1] < 2:
            return 1.0

        N = D.shape[1]  # jumlah trace

        # Total amplitude seluruh sample
        s = np.sum(D)

        # Total energi seluruh sample
        energy = np.sum(D * D)

        # Hindari numerical problems
        if energy <= 1e-12:
            return 0.0

        # Semblance Taner
        semb = (s * s) / (N * energy)

        # Clamp nilai ke [0,1]
        if semb < 0.0:
            semb = 0.0
        elif semb > 1.0:
            semb = 1.0

        return float(semb)
