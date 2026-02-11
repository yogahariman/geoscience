from geosc.seismic.segy import get_segy_trace_headers
from geosc.seismic.plotting.plot_seismic_3d import SeismicPlot3D

segyfile = "/Drive/D/Temp/cluster_labels_3d.sgy"

headers = get_segy_trace_headers(
    segyfile,
    {
        "INLINE": (5, "int32"),
        "XLINE":  (21, "int32"),
    }
)

plotter = SeismicPlot3D(
    segyfile,
    inline=headers["INLINE"],
    xline=headers["XLINE"],
)

# plot inline slice
plotter.plot_inline(350, plot_type="cluster", clip_percentile=99)

# # plot xline slice
# plotter.plot_xline(500, plot_type="cluster", clip_percentile=99)
