import numpy as np
import segyio
from collections import defaultdict


class WindowAttribute:
    """
    Base class for window-based seismic attributes.

    Design principles
    -----------------
    - NO header parsing here
    - NO knowledge about byte position / dtype
    - Geometry (inline/xline) MUST be provided explicitly for 3D

    Window definition
    -----------------
    - 2D: window = (wz, wx)
    - 3D: window = (wz, wx, wy)

    Edge handling
    -------------
    - NO padding
    - NO skipping
    - Window is CLIPPED at boundaries
    """

    def __init__(
        self,
        input_segy: str,
        output_segy: str,
        window: tuple,
        inline: np.ndarray | None = None,
        xline: np.ndarray | None = None,
        load_to_ram: bool = False,   # <--- NEW OPTION
    ):
        self.input_segy = input_segy
        self.output_segy = output_segy
        self.window = window

        self.inline = inline
        self.xline = xline

        self.load_to_ram = load_to_ram     # <---
        self._cube = None                  # <---

        if len(window) not in (2, 3):
            raise ValueError(
                "window must be (wz, wx) for 2D or (wz, wx, wy) for 3D"
            )

        if len(window) == 3:
            if inline is None or xline is None:
                raise ValueError(
                    "3D window requires explicit inline and xline arrays"
                )

    # ==========================================================
    # RAM PRELOAD
    # ==========================================================
    def _preload(self, src):
        """
        Load seluruh trace SEG-Y ke RAM sebagai array 2D:
        shape = (ntraces, nz)
        """
        ntr = src.tracecount
        nz = len(src.samples)

        cube = np.zeros((ntr, nz), dtype=np.float32)
        for i in range(ntr):
            cube[i] = src.trace[i]

        return cube

    # ==========================================================
    # API TO BE IMPLEMENTED BY CHILD CLASS
    # ==========================================================
    def compute(self, D: np.ndarray) -> float:
        """
        Compute attribute value from window matrix D.

        Parameters
        ----------
        D : np.ndarray
            2D array:
            - 2D attribute: (nz_window, nx_window)
            - 3D attribute: (nz_window, ntraces_window)

        Returns
        -------
        float
            Attribute value at current sample
        """
        raise NotImplementedError

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self):
        if len(self.window) == 2:
            self._run_2d()
        else:
            self._run_3d()

    # ==========================================================
    # 2D WINDOW (TRACE-ORDER BASED)
    # ==========================================================
    def _run_2d(self):
        wz, wx = self.window

        with segyio.open(self.input_segy, "r", ignore_geometry=True) as src:

            # ---- preload ke RAM (opsional) ----
            if self.load_to_ram:
                self._cube = self._preload(src)

            nz = len(src.samples)
            ntr = src.tracecount

            # ----------------------------------------------
            # output SEG-Y
            # ----------------------------------------------
            spec = segyio.spec()
            spec.samples = src.samples
            spec.format = src.format
            spec.tracecount = ntr

            with segyio.create(self.output_segy, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin

                for i in range(ntr):
                    out = np.ones(nz, dtype=np.float32)

                    # lateral bounds (CLIPPED)
                    xmin = max(0, i - wx)
                    xmax = min(ntr, i + wx + 1)

                    # ===== RAM vs I/O =====
                    if self.load_to_ram:
                        traces = self._cube[xmin:xmax]
                    else:
                        traces = np.array(
                            [src.trace[j] for j in range(xmin, xmax)],
                            dtype=np.float32,
                        )

                    for iz in range(nz):
                        zmin = max(0, iz - wz)
                        zmax = min(nz, iz + wz + 1)

                        D = traces[:, zmin:zmax].T
                        if D.size == 0:
                            continue

                        out[iz] = self.compute(D)

                    dst.trace[i] = out
                    dst.header[i] = src.header[i]

    # ==========================================================
    # 3D WINDOW (GEOMETRY-DRIVEN)
    # ==========================================================
    def _run_3d(self):
        wz, wx, wy = self.window

        inline = self.inline
        xline = self.xline

        if inline.ndim != 1 or xline.ndim != 1:
            raise ValueError("inline and xline must be 1D arrays")

        if inline.size != xline.size:
            raise ValueError("inline and xline must have same length")

        with segyio.open(self.input_segy, "r", ignore_geometry=True) as src:

            # ---- preload ke RAM (opsional) ----
            if self.load_to_ram:
                self._cube = self._preload(src)

            nz = len(src.samples)
            ntr = src.tracecount

            if ntr != inline.size:
                raise ValueError(
                    "inline/xline size does not match number of traces"
                )

            # --------------------------------------------------
            # build geometry index: index[iline][xline] = trace_id
            # --------------------------------------------------
            index = defaultdict(dict)
            for ti, (il, xl) in enumerate(zip(inline, xline)):
                index[int(il)][int(xl)] = ti

            uniq_il = np.unique(inline)
            uniq_xl = np.unique(xline)

            # --------------------------------------------------
            # output SEG-Y
            # --------------------------------------------------
            spec = segyio.spec()
            spec.samples = src.samples
            spec.format = src.format
            spec.tracecount = ntr

            with segyio.create(self.output_segy, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin

                for il in uniq_il:
                    il = int(il)
                    for xl in uniq_xl:
                        xl = int(xl)

                        if il not in index or xl not in index[il]:
                            continue

                        out = np.ones(nz, dtype=np.float32)

                        for iz in range(nz):
                            zmin = max(0, iz - wz)
                            zmax = min(nz, iz + wz + 1)

                            cols = []

                            # ==== CLIPPED INLINE / XLINE WINDOW ====
                            ilmin = max(il - wy, uniq_il[0])
                            ilmax = min(il + wy, uniq_il[-1])
                            xlmin = max(xl - wx, uniq_xl[0])
                            xlmax = min(xl + wx, uniq_xl[-1])

                            # Pastikan integer Python, bukan numpy.float64
                            ilmin = int(ilmin)
                            ilmax = int(ilmax)
                            xlmin = int(xlmin)
                            xlmax = int(xlmax)

                            cols = []

                            for ilw in range(ilmin, ilmax + 1):
                                for xlw in range(xlmin, xlmax + 1):
                                    ti = index.get(ilw, {}).get(xlw)
                                    if ti is None:
                                        continue

                                    if self.load_to_ram:
                                        cols.append(self._cube[ti][zmin:zmax])
                                    else:
                                        cols.append(src.trace[ti][zmin:zmax])

                            if not cols:
                                continue

                            D = np.array(cols, dtype=np.float32).T
                            out[iz] = self.compute(D)

                        ti0 = index[il][xl]
                        dst.trace[ti0] = out
                        dst.header[ti0] = src.header[ti0]
