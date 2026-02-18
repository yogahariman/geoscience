import numpy as np
import matplotlib.pyplot as plt
from geosc.seismic.plotting.lithology_utils import (
    build_discrete_style,
    mask_null_values,
)

from geosc.seismic.segy import get_segy_trace_headers
from geosc.seismic.plotting.plot_seismic_3d import SeismicPlot3D


segyfile = "/Drive/D/Temp/lithology_pred_3d.sgy"
null_value = -999.25

# warna class bisa Anda atur bebas
# contoh: merah, hijau, kuning, biru, hitam
CLASS_COLORS = {
    1: "green",
    2: "yellow",
    3: "black",
    4: "blue",
    5: "red",
}

# pilih slice yang mau diplot
INLINE_TO_PLOT = 350
# XLINE_TO_PLOT = 500  # aktifkan jika ingin plot xline

def _plot_slice(plotter, data, xaxis, title):
    plot_data = mask_null_values(data, null_value)
    valid = np.asarray(plot_data.compressed(), dtype=float)
    if valid.size == 0:
        raise ValueError(f"Tidak ada nilai lithology valid untuk slice: {title}")

    cmap, norm, classes = build_discrete_style(valid, CLASS_COLORS)

    plt.figure(figsize=(14, 6))
    img = plt.imshow(
        plot_data.T,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[xaxis.min(), xaxis.max(), plotter.ns * plotter.dt, 0],
    )

    cbar = plt.colorbar(img, ticks=classes)
    cbar.set_label("Lithology Class")

    plt.xlabel("Line")
    plt.ylabel("Time (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


headers = get_segy_trace_headers(
    segyfile,
    {
        "INLINE": (5, "int32"),
        "XLINE": (21, "int32"),
    },
)

plotter = SeismicPlot3D(
    segyfile,
    inline=headers["INLINE"],
    xline=headers["XLINE"],
)

# plot inline slice
idata, xlines = plotter.slice_inline(INLINE_TO_PLOT)
_plot_slice(plotter, idata, xlines, f"Lithology 3D - Inline {INLINE_TO_PLOT}")

# contoh plot xline slice
# xdata, ilines = plotter.slice_xline(XLINE_TO_PLOT)
# _plot_slice(plotter, xdata, ilines, f"Lithology 3D - Xline {XLINE_TO_PLOT}")
