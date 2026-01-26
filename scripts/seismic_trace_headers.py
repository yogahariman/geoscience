from geosc.seismic.segy import (get_segy_trace_header, get_segy_trace_headers)
import segyio

headers = get_segy_trace_headers(
    "/Drive/D/out.sgy",
    {
        "INLINE": (25, "int32"),
        "XLINE":  (29, "int32"),
        "CDP":    (21, "int32"),
    }
)

inline = headers["INLINE"]
xline  = headers["XLINE"]
cdp    = headers["CDP"]