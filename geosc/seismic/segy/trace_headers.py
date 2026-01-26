import segyio
import struct
import numpy as np


def get_segy_trace_header(
    segyfile,
    byte_pos,        # Petrel byte number (1-based)
    fmt="int32"      # int8, int16, int32, float32, uint32 (IBM float)
):
    """
    Read ONE trace header field for ALL traces using Petrel-style definition.

    Parameters
    ----------
    segyfile : str
        Path to SEG-Y file
    byte_pos : int
        Byte position in trace header (1-based, Petrel convention)
    fmt : str
        Data format:
        - int8     : 1-byte two's complement
        - int16    : 2-byte two's complement
        - int32    : 4-byte two's complement
        - float32 : 4-byte IEEE floating point
        - uint32  : 4-byte IBM floating point (Petrel style)

    Returns
    -------
    np.ndarray
        Header values for all traces

    examples
    --------
    cdp    = get_segy_trace_header("data.sgy", 21, "int32")
    inline = get_segy_trace_header("data.sgy", 25, "int32")
    xline  = get_segy_trace_header("data.sgy", 29, "int32")
    """

    fmt_map = {
        "int8":    (">b", 1),
        "int16":   (">h", 2),
        "int32":   (">i", 4),
        "float32": (">f", 4),
        "uint32":  ("IBM", 4),   # 👈 IBM floating point
    }

    if fmt not in fmt_map:
        raise ValueError(f"Unsupported format: {fmt}")

    kind, nbyte = fmt_map[fmt]

    # ambil info trace lewat segyio (AMAN)
    with segyio.open(segyfile, "r", ignore_geometry=True) as f:
        ns = f.samples.size
        ntr = f.tracecount

    values = np.empty(ntr, dtype=np.float64)

    # baca byte mentah (Petrel/MATLAB style)
    with open(segyfile, "rb") as fh:
        for tr in range(ntr):
            postBit = 3600 + tr * (240 + ns * 4)
            offset = postBit + (byte_pos - 1)

            fh.seek(offset)
            raw = fh.read(nbyte)

            if kind == "IBM":
                # IBM float → IEEE float
                val = segyio.tools.ibm2ieee(raw)[0]
            else:
                val = struct.unpack(kind, raw)[0]

            values[tr] = val

    return values

import segyio
import struct
import numpy as np


def get_segy_trace_headers(segyfile, header_map):
    """
    header_map example:
    headers = get_segy_trace_headers(
        "data.sgy",
        {
            "INLINE": (25, "int32"),
            "XLINE":  (29, "int32"),
            "OFFSET": (37, "uint32"),   # IBM float
            "ANGLE":  (41, "float32"),
            "FLAG":   (85, "int8"),
        }
    )
    INLINE = headers["INLINE"]
    XLINE  = headers["XLINE"]
    OFFSET = headers["OFFSET"]
    ANGLE  = headers["ANGLE"]
    FLAG   = headers["FLAG"]
    """

    fmt_map = {
        "int8":    (">b", 1),
        "int16":   (">h", 2),
        "int32":   (">i", 4),
        "float32": (">f", 4),
        "uint32":  ("IBM", 4),   # 👈 Petrel-style IBM float
    }

    with segyio.open(segyfile, "r", ignore_geometry=True) as f:
        ns = f.samples.size
        ntr = f.tracecount

    out = {k: np.empty(ntr, dtype=np.float64) for k in header_map}

    with open(segyfile, "rb") as fh:
        for tr in range(ntr):
            postBit = 3600 + tr * (240 + ns * 4)

            for name, (byte_pos, fmt) in header_map.items():
                kind, nbyte = fmt_map[fmt]
                offset = postBit + (byte_pos - 1)
                fh.seek(offset)
                raw = fh.read(nbyte)

                if kind == "IBM":
                    # IBM float → IEEE float
                    val = segyio.tools.ibm2ieee(raw)[0]
                else:
                    val = struct.unpack(kind, raw)[0]

                out[name][tr] = val

    return out
