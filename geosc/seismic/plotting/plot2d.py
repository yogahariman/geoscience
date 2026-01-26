# seismic/plotting/plot_seismic_2d.py

import numpy as np
import segyio
import matplotlib.pyplot as plt


class SeismicPlot2D:
    """
    Simple & memory-safe seismic 2D plotter
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
        Load full or partial seismic section
        """
        if trace_range is None:
            start, end = 0, self.n_traces
        else:
            start, end = trace_range

        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            data = np.stack([
                f.trace[i] for i in range(start, end)
            ])

        return data

    # --------------------------------------------------
    def plot(
        self,
        trace_range=None,
        clip=None,
        cmap="seismic",
        title="Seismic 2D",
        colorbar=True
    ):
        """
        Generic 2D seismic plot
        """
        data = self._load_section(trace_range)

        if clip is not None:
            vmin, vmax = clip
        else:
            vmin, vmax = None, None

        tmax = self.ns * self.dt

        plt.figure(figsize=(14, 6))
        plt.imshow(
            data.T,
            cmap=cmap,
            aspect="auto",
            extent=[
                0, data.shape[0],
                tmax, 0
            ],
            vmin=vmin,
            vmax=vmax
        )

        if colorbar:
            plt.colorbar()

        plt.xlabel("Trace")
        plt.ylabel("Time (ms)")
        plt.title(title)
        plt.show()

    # --------------------------------------------------
    def plot_coherence(
        self,
        trace_range=None,
        clip=(0, 1),
        cmap="viridis"
    ):
        """
        Preset khusus coherence 2D
        """
        self.plot(
            trace_range=trace_range,
            clip=clip,
            cmap=cmap,
            title="Coherence 2D"
        )

    # --------------------------------------------------
    def plot_amplitude(
        self,
        trace_range=None,
        clip_percentile=99
    ):
        """
        Preset amplitude (auto clip)
        """
        data = self._load_section(trace_range)

        vmax = np.percentile(np.abs(data), clip_percentile)
        vmin = -vmax

        tmax = self.ns * self.dt

        plt.figure(figsize=(14, 6))
        plt.imshow(
            data.T,
            cmap="seismic",
            aspect="auto",
            extent=[
                0, data.shape[0],
                tmax, 0
            ],
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar()
        plt.xlabel("Trace")
        plt.ylabel("Time (ms)")
        plt.title("Amplitude 2D")
        plt.show()
