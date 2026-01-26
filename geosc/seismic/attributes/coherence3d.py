from .base.window import WindowAttribute
import numpy as np


class Coherence3D(WindowAttribute):
    """
    3D coherence attribute.
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
        load_to_ram=False,   # opsional, supaya bisa preload cube
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
        """
        # Jika data kosong atau hanya 1 trace → coherence = 1 (definisi)
        if D.size == 0 or D.shape[1] < 2:
            return 1.0

        # G = D^T * D (Covariance matrix)
        G = np.dot(D.T, D)

        # Gunakan eigvalsh untuk matrix simetris (lebih cepat & stabil)
        eig = np.linalg.eigvalsh(G)

        s = eig.sum()

        # Hindari numeric issue (sangat kecil / nol)
        if s <= 1e-12:
            return 1.0

        # coherence = max_eigenvalue / sum(eigenvalues)
        return float(eig[-1] / s)
