from geosc.seismic.attributes import Coherence3D
from geosc.seismic.segy import get_segy_trace_headers
from geosc.seismic.plotting.plot_seismic_3d import SeismicPlot3D

fileIn = "/Drive/D/Works/DataSample/Seismic3D/Sample01/SE-WALIO-PURE-IRIS.sgy"
fileOut = "/Drive/D/coherence_3d.sgy"

headers = get_segy_trace_headers(
    fileIn,
    {
        "INLINE": (25, "int32"),
        "XLINE":  (29, "int32"),
    }
)

attr = Coherence3D(
    fileIn,
    fileOut,
    window=(20, 5, 5),
    inline=headers["INLINE"],
    xline=headers["XLINE"],
    load_to_ram=True
)

attr.run()

plotter = SeismicPlot3D(
    fileOut,
    inline=headers["INLINE"],
    xline=headers["XLINE"],
)

plotter.plot_inline(200, plot_type="coherence")

