from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

from geosc.seismic.segy import get_segy_trace_headers


@dataclass
class HeaderBytes:
    x: Tuple[int, str]
    y: Tuple[int, str]
    inline: Tuple[int, str]
    xline: Tuple[int, str]


def _validate_horizon_array(horizon_xyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(horizon_xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Horizon array must be Nx3 [x, y, z/time].")
    if arr.shape[0] == 0:
        raise ValueError("Horizon array is empty.")
    return arr


class SeismicMLBase:
    """Shared seismic ML foundation for alignment, horizon handling, and sample-axis config."""

    def __init__(
        self,
        input_segy_list: Union[List[str], List[List[str]]],
        output_segy: Union[str, List[str]],
        horizon_top: np.ndarray,
        horizon_base: np.ndarray,
        header_bytes: List[Dict[str, Tuple[int, str]]],
        null_value: float = -999.25,
        t0: Union[float, List, None] = None,
        dt: Union[float, List, None] = None,
        ns: Union[int, List, None] = None,
    ) -> None:
        if not input_segy_list:
            raise ValueError("input_segy_list must contain at least one SEG-Y path.")

        required_base = {"X", "Y"}
        input_is_multiline = isinstance(input_segy_list[0], list)
        if input_is_multiline:
            n_lines = len(input_segy_list)
            n_attrs = len(input_segy_list[0])
            for li in range(n_lines):
                if len(input_segy_list[li]) != n_attrs:
                    raise ValueError("All line lists must have same nAttributes.")
        else:
            n_attrs = len(input_segy_list)
            n_lines = 1

        if input_is_multiline:
            if len(header_bytes) != n_lines:
                raise ValueError("header_bytes must match number of lines.")
            for li, hb_line in enumerate(header_bytes):
                if len(hb_line) != n_attrs:
                    raise ValueError("header_bytes line length must match nAttributes.")
                for ai, hb in enumerate(hb_line):
                    keys = {k.upper() for k in hb}
                    missing = required_base.difference(keys)
                    if missing:
                        raise ValueError(
                            f"header_bytes[{li}][{ai}] missing keys: {sorted(missing)}"
                        )
                    if not ({"INLINE", "XLINE"} <= keys or "CDP" in keys):
                        raise ValueError(
                            f"header_bytes[{li}][{ai}] must include INLINE+XLINE (3D) or CDP (2D)."
                        )
        else:
            if len(header_bytes) != n_attrs:
                raise ValueError("header_bytes must match number of attributes.")
            for idx, hb in enumerate(header_bytes):
                keys = {k.upper() for k in hb}
                missing = required_base.difference(keys)
                if missing:
                    raise ValueError(
                        f"header_bytes[{idx}] missing keys: {sorted(missing)}"
                    )
                if not ({"INLINE", "XLINE"} <= keys or "CDP" in keys):
                    raise ValueError(
                        f"header_bytes[{idx}] must include INLINE+XLINE (3D) or CDP (2D)."
                    )

        self.input_segy_list = input_segy_list
        self.output_segy = output_segy
        self.horizon_top = _validate_horizon_array(horizon_top)
        self.horizon_base = _validate_horizon_array(horizon_base)
        if input_is_multiline:
            self.header_bytes = [
                [
                    {k.upper(): v for k, v in hb.items()} for hb in hb_line
                ]
                for hb_line in header_bytes
            ]
        else:
            self.header_bytes = [
                {k.upper(): v for k, v in hb.items()} for hb in header_bytes
            ]
        self.null_value = float(null_value)

        t0_values = None
        if t0 is not None:
            if input_is_multiline:
                if isinstance(t0, (int, float)):
                    t0_values = [
                        [float(t0) for _ in range(n_attrs)]
                        for _ in range(n_lines)
                    ]
                else:
                    t0_values = t0  # type: ignore[assignment]
            else:
                if isinstance(t0, (int, float)):
                    t0_values = [float(t0) for _ in range(n_attrs)]
                else:
                    t0_values = t0  # type: ignore[assignment]

        if t0_values is None:
            if input_is_multiline:
                self.t0 = [
                    [0.0 for _ in range(n_attrs)] for _ in range(n_lines)
                ]
            else:
                self.t0 = [0.0 for _ in range(n_attrs)]
        else:
            if input_is_multiline:
                if len(t0_values) != n_lines:
                    raise ValueError("t0 must match number of lines.")
                for li in range(n_lines):
                    if len(t0_values[li]) != n_attrs:
                        raise ValueError(
                            "t0 line length must match nAttributes."
                        )
                self.t0 = [
                    [float(v) for v in line] for line in t0_values
                ]
            else:
                if len(t0_values) != n_attrs:
                    raise ValueError("t0 must match number of attributes.")
                self.t0 = [float(v) for v in t0_values]

        def _normalize_axis_param(
            raw: Union[float, int, List, None],
            *,
            name: str,
            cast_fn,
        ):
            if raw is None:
                if input_is_multiline:
                    return [[None for _ in range(n_attrs)] for _ in range(n_lines)]
                return [None for _ in range(n_attrs)]

            if np.isscalar(raw):
                value = cast_fn(raw)
                if input_is_multiline:
                    return [[value for _ in range(n_attrs)] for _ in range(n_lines)]
                return [value for _ in range(n_attrs)]

            if input_is_multiline:
                if len(raw) != n_lines:  # type: ignore[arg-type]
                    raise ValueError(f"{name} must match number of lines.")
                out = []
                for li in range(n_lines):
                    line_raw = raw[li]  # type: ignore[index]
                    if len(line_raw) != n_attrs:
                        raise ValueError(f"{name} line length must match nAttributes.")
                    out.append(
                        [None if v is None else cast_fn(v) for v in line_raw]
                    )
                return out

            if len(raw) != n_attrs:  # type: ignore[arg-type]
                raise ValueError(f"{name} must match number of attributes.")
            return [None if v is None else cast_fn(v) for v in raw]  # type: ignore[return-value]

        self._n_attrs = n_attrs
        self._n_lines = n_lines
        self._input_is_multiline = input_is_multiline
        self.dt = _normalize_axis_param(dt, name="dt", cast_fn=float)
        self.ns = _normalize_axis_param(ns, name="ns", cast_fn=int)

    def _read_headers(self, segyfile: str, hb: Dict[str, Tuple[int, str]]) -> Dict[str, np.ndarray]:
        header_map = {
            "X": hb["X"],
            "Y": hb["Y"],
        }
        if "INLINE" in hb and "XLINE" in hb:
            header_map["INLINE"] = hb["INLINE"]
            header_map["XLINE"] = hb["XLINE"]
        if "CDP" in hb:
            header_map["CDP"] = hb["CDP"]
        return get_segy_trace_headers(segyfile, header_map)

    def _get_line_inputs(self, line_idx: int) -> Tuple[List[str], str]:
        if self._input_is_multiline:
            segys = list(self.input_segy_list[line_idx])  # type: ignore[index]
            if isinstance(self.output_segy, list):
                out_path = self.output_segy[line_idx]
            else:
                raise ValueError("output_segy must be list when using multiline input.")
            return segys, out_path
        segys = list(self.input_segy_list)  # type: ignore[arg-type]
        if isinstance(self.output_segy, list):
            raise ValueError("output_segy must be str when using single line input.")
        return segys, self.output_segy  # type: ignore[return-value]

    @staticmethod
    def _trace_keys(headers: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
        if "INLINE" in headers and "XLINE" in headers:
            return [
                (int(il), int(xl))
                for il, xl in zip(headers["INLINE"], headers["XLINE"])
            ]
        if "CDP" in headers:
            return [
                (int(cdp), 0)
                for cdp in headers["CDP"]
            ]
        raise ValueError("Header missing INLINE/XLINE or CDP.")

    @staticmethod
    def _build_horizon_tree(horizon_xyz: np.ndarray) -> Tuple[cKDTree, np.ndarray]:
        xy = horizon_xyz[:, :2]
        t = horizon_xyz[:, 2]
        return cKDTree(xy), t

    @staticmethod
    def _nearest_time(
        tree: cKDTree,
        times: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        dist, idx = tree.query(np.c_[x, y], k=1)
        _ = dist
        return times[idx]

    @staticmethod
    def _time_to_index(samples: np.ndarray, t: float) -> int:
        return int(np.argmin(np.abs(samples - t)))

    @staticmethod
    def _build_index_map(samples_ref: np.ndarray, samples_target: np.ndarray) -> np.ndarray:
        return np.array(
            [int(np.argmin(np.abs(samples_target - t))) for t in samples_ref],
            dtype=int,
        )

    @staticmethod
    def _build_samples_axis(
        src,
        t0: float,
        dt: float | None,
        ns: int | None,
    ) -> np.ndarray:
        if dt is not None:
            n = int(ns) if ns is not None else int(src.samples.size)
            return t0 + (np.arange(n, dtype=float) * float(dt))
        samples = np.asarray(src.samples, dtype=float) + t0
        if ns is not None and int(ns) != samples.size:
            raise ValueError("ns does not match SEG-Y samples size.")
        return samples

    def _build_alignment(
        self,
        headers_list: List[Dict[str, np.ndarray]],
    ) -> Tuple[List[int], List[List[int]], Dict[int, int], np.ndarray, np.ndarray]:
        ref_headers = headers_list[0]

        mappings: List[Dict[Tuple[int, int], int]] = []
        for h in headers_list:
            keys = self._trace_keys(h)
            mappings.append({k: i for i, k in enumerate(keys)})

        intersect_keys = set(mappings[0].keys())
        for m in mappings[1:]:
            intersect_keys &= set(m.keys())
        if not intersect_keys:
            raise ValueError("No intersecting X/Y keys across attributes.")

        ref_keys = self._trace_keys(ref_headers)
        aligned_keys = [k for k in ref_keys if k in intersect_keys]
        aligned_ref_indices = [mappings[0][k] for k in aligned_keys]
        aligned_trace_indices_per_volume = [
            [m[k] for k in aligned_keys] for m in mappings
        ]
        ref_idx_to_j = {int(ref_idx): j for j, ref_idx in enumerate(aligned_ref_indices)}

        ref_x = ref_headers["X"][aligned_ref_indices]
        ref_y = ref_headers["Y"][aligned_ref_indices]
        return aligned_ref_indices, aligned_trace_indices_per_volume, ref_idx_to_j, ref_x, ref_y

    def _build_interval_index_maps(
        self,
        aligned_ref_indices: List[int],
        top_t: np.ndarray,
        base_t: np.ndarray,
        samples_ref: np.ndarray,
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        top_idx_map: Dict[int, int] = {}
        base_idx_map: Dict[int, int] = {}
        for ref_idx, t_top, t_base in zip(aligned_ref_indices, top_t, base_t):
            if not np.isfinite(t_top) or not np.isfinite(t_base):
                continue
            iz_top = self._time_to_index(samples_ref, float(t_top))
            iz_base = self._time_to_index(samples_ref, float(t_base))
            if iz_top > iz_base:
                iz_top, iz_base = iz_base, iz_top
            top_idx_map[int(ref_idx)] = int(iz_top)
            base_idx_map[int(ref_idx)] = int(iz_base)
        return top_idx_map, base_idx_map
