from .base.window import WindowAttribute
import numpy as np


class Semblance2D(WindowAttribute):
    """
    2D semblance attribute.
    Window = (wz, wx)
    """

    def __init__(self, input_segy, output_segy, window, load_to_ram=False):
        super().__init__(
            input_segy=input_segy,
            output_segy=output_segy,
            window=window,
            load_to_ram=load_to_ram,
        )

    def compute(self, D: np.ndarray) -> float:
        """
        D shape: (nz_window, ntraces_window)
        Semblance formula (Taner):
            S = ( (sum over all samples)^2 ) / ( N * sum of squares )
        """

        # Jika window kosong atau hanya 1 trace → semblance = 1
        if D.size == 0 or D.shape[1] < 2:
            return 1.0

        # N = jumlah trace pada window
        N = D.shape[1]

        # Sum seluruh sample pada semua trace
        s = np.sum(D)

        # Sum of squares
        energy = np.sum(D * D)

        if energy <= 1e-12:
            return 0.0

        # Taner semblance
        semb = (s * s) / (N * energy)

        # Pastikan tidak keluar interval [0,1]
        if semb < 0.0:
            semb = 0.0
        elif semb > 1.0:
            semb = 1.0

        return float(semb)
