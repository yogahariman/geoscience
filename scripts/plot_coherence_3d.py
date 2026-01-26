from geosc.seismic.plotting import SeismicPlot3D
from geosc.seismic.segy.trace_headers import get_segy_trace_headers

fileIn = "/Drive/D/coherence_3d.sgy"

headers = get_segy_trace_headers(
    fileIn,
    {
        "INLINE": (25, "int32"),
        "XLINE":  (29, "int32"),
    }
)

plotter = SeismicPlot3D(
    fileIn,
    inline=headers["INLINE"],
    xline=headers["XLINE"],
)

plotter.plot_inline(200, plot_type="coherence")
