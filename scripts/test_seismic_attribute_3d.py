import numpy as np
from geosc.seismic.attributes import InstantaneousAmplitude
from geosc.seismic.plotting import SeismicPlot3D
from geosc.seismic.segy import append_process_text_header
from geosc.seismic.segy.trace_headers import get_segy_trace_headers

fileIn = "/Drive/D/Works/DataSample/Seismic3D/Sample01/SE-WALIO-PURE-IRIS.sgy"
fileOut = "/Drive/D/out.sgy"

InstantaneousAmplitude(fileIn, fileOut).run()
append_process_text_header(
    fileOut,
    process_name="InstantaneousAmplitude",
    details=[
        f"input={fileIn}",
        f"output={fileOut}",
    ],
)

headers = get_segy_trace_headers(
    fileIn,
    {
        "INLINE": (25, "int32"),
        "XLINE":  (29, "int32"),
    }
)

plotter = SeismicPlot3D(
    fileOut,
    inline=headers["INLINE"],
    xline=headers["XLINE"],
)

plotter.plot_inline(200, plot_type="amplitude")
