from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import segyio

from geosc.seismic.segy import get_segy_trace_headers


def resample_segy_3d(
    input_segy: str,
    output_segy: str,
    inline_step: int = 2,
    xline_step: int = 2,
    header_bytes: Dict[str, Tuple[int, str]] | None = None,
) -> None:
    """
    Resample 3D SEG-Y laterally using INLINE/XLINE decimation.

    This function keeps every N-th unique INLINE and every M-th unique XLINE.
    Z axis (samples/time) is preserved as-is.

    Parameters
    ----------
    input_segy : str
        Input SEG-Y path.
    output_segy : str
        Output SEG-Y path.
    inline_step : int, default=2
        Keep every ``inline_step`` unique inline.
    xline_step : int, default=2
        Keep every ``xline_step`` unique xline.
    header_bytes : dict[str, tuple[int, str]] | None
        Header mapping for geometry extraction.
        Default: {"INLINE": (25, "int32"), "XLINE": (29, "int32")}.
    """
    if int(inline_step) < 1 or int(xline_step) < 1:
        raise ValueError("inline_step and xline_step must be >= 1.")

    hb = {"INLINE": (25, "int32"), "XLINE": (29, "int32")}
    if header_bytes is not None:
        hb = {k.upper(): v for k, v in header_bytes.items()}
    if "INLINE" not in hb or "XLINE" not in hb:
        raise ValueError("header_bytes must include INLINE and XLINE.")

    headers = get_segy_trace_headers(
        input_segy,
        {"INLINE": hb["INLINE"], "XLINE": hb["XLINE"]},
    )
    inline = np.asarray(headers["INLINE"], dtype=int)
    xline = np.asarray(headers["XLINE"], dtype=int)

    uniq_il = np.unique(inline)
    uniq_xl = np.unique(xline)
    keep_il = set(uniq_il[:: int(inline_step)].tolist())
    keep_xl = set(uniq_xl[:: int(xline_step)].tolist())

    keep_mask = np.array(
        [(int(il) in keep_il) and (int(xl) in keep_xl) for il, xl in zip(inline, xline)],
        dtype=bool,
    )
    trace_idx = np.where(keep_mask)[0]
    if trace_idx.size == 0:
        raise ValueError("No traces selected after inline/xline resampling.")

    with segyio.open(input_segy, "r", ignore_geometry=True) as src:
        spec = segyio.spec()
        spec.samples = src.samples
        spec.format = src.format
        spec.tracecount = int(trace_idx.size)

        with segyio.create(output_segy, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin

            for i_out, i_src in enumerate(trace_idx):
                dst.trace[i_out] = np.asarray(src.trace[int(i_src)], dtype=np.float32)
                dst.header[i_out] = src.header[int(i_src)]
