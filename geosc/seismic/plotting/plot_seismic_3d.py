import numpy as np
import segyio
import matplotlib.pyplot as plt


class SeismicPlot3D:
    """
    3D Seismic plotter with inline/xline slicing.

    Design principles
    -----------------
    - NO header parsing here
    - Geometry (inline/xline) must be provided explicitly
    - Safe for Petrel / non-standard SEG-Y

    Supports:
    - amplitude
    - coherence
    - semblance
    """

    def __init__(
        self,
        segyfile: str,
        inline: np.ndarray,
        xline: np.ndarray,
    ):
        self.segyfile = segyfile
        self.inline = inline
        self.xline = xline

        if inline.ndim != 1 or xline.ndim != 1:
            raise ValueError("inline and xline must be 1D arrays")

        if inline.size != xline.size:
            raise ValueError("inline and xline must have same length")

        with segyio.open(segyfile, "r", ignore_geometry=True) as f:
            if f.tracecount != inline.size:
                raise ValueError(
                    "inline/xline size does not match number of traces"
                )

            self.n_traces = f.tracecount
            self.ns = f.samples.size
            self.dt = segyio.tools.dt(f) / 1000.0  # ms

    # --------------------------------------------------
    def _load_traces(self, indices):
        with segyio.open(self.segyfile, "r", ignore_geometry=True) as f:
            return np.stack([f.trace[i] for i in indices])

    # --------------------------------------------------
    def slice_inline(self, iline):
        idx = np.where(self.inline == iline)[0]
        if idx.size == 0:
            raise ValueError(f"Inline {iline} tidak ditemukan")
        return self._load_traces(idx), self.xline[idx]

    # --------------------------------------------------
    def slice_xline(self, xline):
        idx = np.where(self.xline == xline)[0]
        if idx.size == 0:
            raise ValueError(f"Xline {xline} tidak ditemukan")
        return self._load_traces(idx), self.inline[idx]

    # ==================================================
    # ===================== PLOT =======================
    # ==================================================
    def plot_slice(
        self,
        data,
        xaxis,
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

        # -----------------------------
        # Default style per plot type
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
            extent=[xaxis.min(), xaxis.max(), tmax, 0],
            vmin=vmin,
            vmax=vmax,
        )

        if colorbar:
            plt.colorbar()

        plt.xlabel("Line")
        plt.ylabel("Time (ms)")
        plt.title(title or plot_type.capitalize())
        plt.show()

    # ==================================================
    # ================= SHORTCUT =======================
    # ==================================================
    def plot_inline(
        self,
        iline,
        plot_type="amplitude",
        clip_percentile=99,
    ):
        data, xlines = self.slice_inline(iline)
        self.plot_slice(
            data,
            xlines,
            plot_type=plot_type,
            clip_percentile=clip_percentile,
            title=f"Inline {iline} ({plot_type})",
        )

    # --------------------------------------------------
    def plot_xline(
        self,
        xline,
        plot_type="amplitude",
        clip_percentile=99,
    ):
        data, ilines = self.slice_xline(xline)
        self.plot_slice(
            data,
            ilines,
            plot_type=plot_type,
            clip_percentile=clip_percentile,
            title=f"Xline {xline} ({plot_type})",
        )
