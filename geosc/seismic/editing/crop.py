from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import segyio

from geosc.seismic.segy import get_segy_trace_headers


def crop_segy_3d(
    input_segy: str,
    output_segy: str,
    inline_range: Tuple[int, int],
    xline_range: Tuple[int, int],
    time_range: Tuple[float, float] | None = None,
    t0: float | None = None,
    header_bytes: Dict[str, Tuple[int, str]] | None = None,
) -> None:
    """
    Crop a 3D SEG-Y volume by inline/xline/time ranges (inclusive).

    Parameters
    ----------
    input_segy : str
        Input SEG-Y path.
    output_segy : str
        Output SEG-Y path.
    inline_range : tuple[int, int]
        Inclusive inline range, e.g. (100, 200).
    xline_range : tuple[int, int]
        Inclusive xline range, e.g. (100, 200).
    time_range : tuple[float, float] | None
        Inclusive time range in milliseconds, e.g. (0.0, 500.0).
        If None, all samples on Z axis are preserved.
    t0 : float | None
        Time of first sample (ms). If provided, time filtering is applied on
        ``samples + t0`` axis.
    header_bytes : dict[str, tuple[int, str]] | None
        Header mapping for geometry extraction.
        Default: {"INLINE": (25, "int32"), "XLINE": (29, "int32")}.
    """
    hb = {"INLINE": (25, "int32"), "XLINE": (29, "int32")}
    if header_bytes is not None:
        hb = {k.upper(): v for k, v in header_bytes.items()}

    if "INLINE" not in hb or "XLINE" not in hb:
        raise ValueError("header_bytes must include INLINE and XLINE.")

    il_min, il_max = sorted((int(inline_range[0]), int(inline_range[1])))
    xl_min, xl_max = sorted((int(xline_range[0]), int(xline_range[1])))
    if time_range is not None:
        t_min, t_max = sorted((float(time_range[0]), float(time_range[1])))

    headers = get_segy_trace_headers(
        input_segy,
        {"INLINE": hb["INLINE"], "XLINE": hb["XLINE"]},
    )

    inline = np.asarray(headers["INLINE"], dtype=int)
    xline = np.asarray(headers["XLINE"], dtype=int)
    trace_mask = (
        (inline >= il_min)
        & (inline <= il_max)
        & (xline >= xl_min)
        & (xline <= xl_max)
    )
    trace_idx = np.where(trace_mask)[0]
    if trace_idx.size == 0:
        raise ValueError("No traces found inside inline/xline ranges.")

    with segyio.open(input_segy, "r", ignore_geometry=True) as src:
        samples = np.asarray(src.samples, dtype=float)
        samples_axis = samples if t0 is None else (samples + float(t0))
        if time_range is None:
            sample_idx = np.arange(samples.size, dtype=int)
        else:
            sample_mask = (samples_axis >= t_min) & (samples_axis <= t_max)
            sample_idx = np.where(sample_mask)[0]
            if sample_idx.size == 0:
                raise ValueError("No samples found inside time range.")

        out_samples = samples_axis[sample_idx]
        ntr_out = int(trace_idx.size)
        ns_out = int(sample_idx.size)

        spec = segyio.spec()
        spec.samples = out_samples
        spec.format = src.format
        spec.tracecount = ntr_out

        with segyio.create(output_segy, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            try:
                dst.bin[segyio.BinField.Samples] = ns_out
            except Exception:
                pass

            for i_out, i_src in enumerate(trace_idx):
                cropped = np.asarray(src.trace[int(i_src)], dtype=np.float32)[sample_idx]
                dst.trace[i_out] = cropped
                dst.header[i_out] = src.header[int(i_src)]
                try:
                    dst.header[i_out][segyio.TraceField.TRACE_SAMPLE_COUNT] = ns_out
                except Exception:
                    pass
