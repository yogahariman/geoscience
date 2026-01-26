from .base.window import WindowAttribute
import numpy as np


class Coherence2D(WindowAttribute):
    """
    2D coherence attribute.
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
        Computes semblance-like coherence using eigen decomposition.
        """
        # Kalau window kecil / data kosong → coherence = 1
        if D.size == 0 or D.shape[1] < 2:
            return 1.0

        # G = D^T D  (matrix covariance)
        G = np.dot(D.T, D)

        # Eigenvalues of symmetric matrix (more stable)
        eig = np.linalg.eigvalsh(G)

        s = eig.sum()

        # Hindari numerical issue (sum sangat kecil → noise)
        if s <= 1e-12:
            return 1.0

        return float(eig[-1] / s)
