import numpy as np
import segyio
import matplotlib.pyplot as plt


class SeismicPlot2D:
    """
    Simple & consistent 2D seismic plotter
    Supports amplitude / coherence / semblance plots
    """

    def __init__(self, segyfile):
        self.segyfile = segyfile

        with segyio.open(segyfile, "r", ignore_geometry=True) as f:
            self.n_traces = f.tracecount
            self.ns = f.samples.size
            self.dt = segyio.tools.dt(f) / 1000.0  # ms

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
        """

        plot_type = plot_type.lower()
        data = self._load_section(trace_range)

        # -----------------------------
        # Style per plot type
        # -----------------------------
        if plot_type == "amplitude":
            vmax = np.percentile(np.abs(data), clip_percentile)
            vmin = -vmax
            cmap = cmap or "seismic"

        elif plot_type in ("coherence", "semblance"):
            vmin, vmax = 0.0, 1.0
            cmap = cmap or "gray_r"

        elif plot_type == "cluster":
            valid = data[data != null_value]
            if valid.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.nanmin(valid))
                vmax = float(np.nanmax(valid))
                if vmin == vmax:
                    vmax = vmin + 1.0
            cmap = cmap or "tab20"

        else:
            raise ValueError(
                "plot_type harus salah satu dari: "
                "'amplitude', 'coherence', 'semblance', 'cluster'"
            )

        tmax = self.ns * self.dt

        plt.figure(figsize=(14, 6))
        plt.imshow(
            data.T,
            aspect="auto",
            cmap=cmap,
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
