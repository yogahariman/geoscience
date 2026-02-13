import numpy as np
import segyio
import matplotlib.pyplot as plt


class SeismicPlot2D:
    """
    Simple & consistent 2D seismic plotter
    Supports amplitude / coherence / semblance / cluster / regression plots
    """

    def __init__(self, segyfile):
        self.segyfile = segyfile

        with segyio.open(segyfile, "r", ignore_geometry=True) as f:
            self.n_traces = f.tracecount
            self.ns = f.samples.size
            self.dt = segyio.tools.dt(f) / 1000.0  # ms

    # --------------------------------------------------
    @staticmethod
    def _mask_null(data, null_value):
        arr = np.asarray(data, dtype=float).copy()
        arr[arr == null_value] = np.nan
        return np.ma.masked_invalid(arr)

    # --------------------------------------------------
    @staticmethod
    def _cmap_with_transparent_bad(cmap):
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad((0, 0, 0, 0))
        return cmap_obj

    # --------------------------------------------------
    def _load_section(self, trace_range=None):
        """
        Load full or partial 2D section
        """
        if trace_range is None:
            start, end = 0, self.n_traces
        else:
            start, end = trace_range

        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            data = np.stack([f.trace[i] for i in range(start, end)])

        return data

    # ==================================================
    # ===================== PLOT =======================
    # ==================================================
    def plot(
        self,
        trace_range=None,
        plot_type="amplitude",
        clip_percentile=99,
        cmap=None,
        title=None,
        colorbar=True,
        null_value=-999.25,
    ):
        """
        plot_type:
            - 'amplitude'
            - 'coherence'
            - 'semblance'
            - 'cluster'
            - 'regression'
            - 'vsh'
        """

        plot_type = plot_type.lower()
        data = self._load_section(trace_range)
        data_plot = self._mask_null(data, null_value)
        valid = np.asarray(data_plot.compressed(), dtype=float)

        # -----------------------------
        # Style per plot type
        # -----------------------------
        if plot_type == "amplitude":
            if valid.size == 0:
                vmax = 1.0
            else:
                vmax = np.percentile(np.abs(valid), clip_percentile)
            vmin = -vmax
            cmap = cmap or "seismic"

        elif plot_type in ("coherence", "semblance"):
            vmin, vmax = 0.0, 1.0
            cmap = cmap or "gray_r"

        elif plot_type == "cluster":
            if valid.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.nanmin(valid))
                vmax = float(np.nanmax(valid))
                if vmin == vmax:
                    vmax = vmin + 1.0
            cmap = cmap or "tab20"

        elif plot_type == "regression":
            if valid.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.nanpercentile(valid, 1.0))
                vmax = float(np.nanpercentile(valid, 99.0))
                if vmin == vmax:
                    vmax = vmin + 1.0
            cmap = cmap or "viridis"

        elif plot_type == "vsh":
            # Vshale convention: 0 (clean) to 1 (shale)
            vmin, vmax = 0.0, 1.0
            cmap = cmap or "rainbow_r"

        else:
            raise ValueError(
                "plot_type harus salah satu dari: "
                "'amplitude', 'coherence', 'semblance', 'cluster', 'regression', 'vsh'"
            )

        tmax = self.ns * self.dt

        plt.figure(figsize=(14, 6))
        plt.imshow(
            data_plot.T,
            aspect="auto",
            cmap=self._cmap_with_transparent_bad(cmap),
            extent=[
                0,
                data.shape[0],
                tmax,
                0
            ],
            vmin=vmin,
            vmax=vmax
        )

        if colorbar:
            plt.colorbar()

        plt.xlabel("Trace")
        plt.ylabel("Time (ms)")
        plt.title(title or f"Seismic 2D ({plot_type})")
        plt.show()

    # ==================================================
    # ================= SHORTCUT =======================
    # ==================================================
    def plot_amplitude(
        self,
        trace_range=None,
        clip_percentile=99
    ):
        self.plot(
            trace_range=trace_range,
            plot_type="amplitude",
            clip_percentile=clip_percentile,
            title="Amplitude 2D"
        )

    # --------------------------------------------------
    def plot_coherence(
        self,
        trace_range=None
    ):
        self.plot(
            trace_range=trace_range,
            plot_type="coherence",
            title="Coherence 2D"
        )

    # --------------------------------------------------
    def plot_semblance(
        self,
        trace_range=None
    ):
        self.plot(
            trace_range=trace_range,
            plot_type="semblance",
            title="Semblance 2D"
        )
