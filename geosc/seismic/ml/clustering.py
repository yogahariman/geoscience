from __future__ import annotations

from dataclasses import dataclass
import shutil
from typing import Dict, List, Tuple, Union

import numpy as np
import segyio
from scipy.spatial import cKDTree

from geosc.ml import Clusterer
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


def _build_horizon_tree(horizon_xyz: np.ndarray) -> Tuple[cKDTree, np.ndarray]:
    xy = horizon_xyz[:, :2]
    t = horizon_xyz[:, 2]
    return cKDTree(xy), t


def _nearest_time(tree: cKDTree, times: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dist, idx = tree.query(np.c_[x, y], k=1)
    _ = dist
    return times[idx]


def _time_to_index(samples: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(samples - t)))


def _build_index_map(samples_ref: np.ndarray, samples_target: np.ndarray) -> np.ndarray:
    """
    Map reference sample indices -> nearest indices in target samples.
    samples_ref/samples_target are 1D arrays in the same unit.
    """
    return np.array(
        [int(np.argmin(np.abs(samples_target - t))) for t in samples_ref],
        dtype=int,
    )


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


class SeismicClusterer:
    """
    Seismic clustering within horizon interval, with multi-volume inputs.

    - Input volumes are aligned by X/Y headers.
    - Horizon top/base are provided as Nx3 [x,y,z/time] and matched by nearest x,y.
    - Training uses random per-sample sampling of the interval (percentage).
    - Output is a SEG-Y volume of cluster labels.
    """

    def __init__(
        self,
        input_segy_list: Union[List[str], List[List[str]]],
        output_segy: Union[str, List[str]],
        horizon_top: np.ndarray,
        horizon_base: np.ndarray,
        header_bytes: List[Dict[str, Tuple[int, str]]],
        sample_percent: float = 100.0,
        null_value: float = -999.25,
        cluster_params: Dict | None = None,
        seed: int | None = None,
        model_type: str = "som",
        time_first_sample: List[float] | None = None,
        seis_time_first_sample: Union[float, List, None] = None,
        dt: float | None = None,
        ns: int | None = None,
        label_offset: int = 0,
        model_output: str | None = None,
    ) -> None:
        if not input_segy_list:
            raise ValueError("input_segy_list must contain at least one SEG-Y path.")
        if sample_percent <= 0 or sample_percent > 100:
            raise ValueError("sample_percent must be in (0, 100].")

        # Header requirements
        # 3D: X,Y,INLINE,XLINE
        # 2D: X,Y,CDP
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
        self.sample_percent = sample_percent
        self.null_value = null_value
        self.cluster_params = cluster_params or {}
        self.seed = seed
        self.model_type = model_type
        self.label_offset = int(label_offset)
        self.model_output = model_output
        if seis_time_first_sample is not None:
            if input_is_multiline:
                if isinstance(seis_time_first_sample, (int, float)):
                    time_first_sample = [
                        [float(seis_time_first_sample) for _ in range(n_attrs)]
                        for _ in range(n_lines)
                    ]
                else:
                    time_first_sample = seis_time_first_sample  # type: ignore[assignment]
            else:
                if isinstance(seis_time_first_sample, (int, float)):
                    time_first_sample = [float(seis_time_first_sample) for _ in range(n_attrs)]
                else:
                    time_first_sample = seis_time_first_sample  # type: ignore[assignment]

        if time_first_sample is None:
            if input_is_multiline:
                self.time_first_sample = [
                    [0.0 for _ in range(n_attrs)] for _ in range(n_lines)
                ]
            else:
                self.time_first_sample = [0.0 for _ in range(n_attrs)]
        else:
            if input_is_multiline:
                if len(time_first_sample) != n_lines:
                    raise ValueError("time_first_sample must match number of lines.")
                for li in range(n_lines):
                    if len(time_first_sample[li]) != n_attrs:
                        raise ValueError(
                            "time_first_sample line length must match nAttributes."
                        )
                self.time_first_sample = [
                    [float(v) for v in line] for line in time_first_sample
                ]
            else:
                if len(time_first_sample) != n_attrs:
                    raise ValueError("time_first_sample must match number of attributes.")
                self.time_first_sample = [float(v) for v in time_first_sample]

        self._n_attrs = n_attrs
        self._n_lines = n_lines
        self._input_is_multiline = input_is_multiline
        self.dt = float(dt) if dt is not None else None
        self.ns = int(ns) if ns is not None else None

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

    def run(self) -> None:
        rng = np.random.RandomState(self.seed)

        # ---- load horizon (matrix Nx3) ----
        top_tree, top_time = _build_horizon_tree(self.horizon_top)
        base_tree, base_time = _build_horizon_tree(self.horizon_base)

        def _get_line_inputs(line_idx: int) -> Tuple[List[str], str]:
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

        X_train: List[List[float]] = []

        for line_idx in range(self._n_lines):
            segy_list, output_path = _get_line_inputs(line_idx)

            # ---- read headers for each attribute ----
            if self._input_is_multiline:
                hb_line = self.header_bytes[line_idx]  # type: ignore[index]
            else:
                hb_line = self.header_bytes
            headers_list = [
                self._read_headers(p, hb) for p, hb in zip(segy_list, hb_line)
            ]
            ref_headers = headers_list[0]

            # ---- build key mapping (3D: INLINE/XLINE, 2D: CDP) ----
            mappings: List[Dict[Tuple[int, int], int]] = []
            for h in headers_list:
                if "INLINE" in h and "XLINE" in h:
                    keys = [
                        (int(il), int(xl))
                        for il, xl in zip(h["INLINE"], h["XLINE"])
                    ]
                elif "CDP" in h:
                    keys = [
                        (int(cdp), 0)
                        for cdp in h["CDP"]
                    ]
                else:
                    raise ValueError("Header missing INLINE/XLINE or CDP.")
                mapping = {k: i for i, k in enumerate(keys)}
                mappings.append(mapping)

            intersect_keys = set(mappings[0].keys())
            for m in mappings[1:]:
                intersect_keys &= set(m.keys())

            if not intersect_keys:
                raise ValueError("No intersecting X/Y keys across attributes.")

            # Preserve reference trace order
            if "INLINE" in ref_headers and "XLINE" in ref_headers:
                ref_keys = [
                    (int(il), int(xl))
                    for il, xl in zip(ref_headers["INLINE"], ref_headers["XLINE"])
                ]
            else:
                ref_keys = [
                    (int(cdp), 0)
                    for cdp in ref_headers["CDP"]
                ]
            aligned_keys = [k for k in ref_keys if k in intersect_keys]
            aligned_ref_indices = [mappings[0][k] for k in aligned_keys]

            aligned_trace_indices_per_volume = [
                [m[k] for k in aligned_keys] for m in mappings
            ]
            ref_idx_to_j = {int(ref_idx): j for j, ref_idx in enumerate(aligned_ref_indices)}

            # ---- horizon matching for aligned traces (XY nearest) ----
            ref_x = ref_headers["X"][aligned_ref_indices]
            ref_y = ref_headers["Y"][aligned_ref_indices]

            top_t = _nearest_time(top_tree, top_time, ref_x, ref_y)
            base_t = _nearest_time(base_tree, base_time, ref_x, ref_y)

            # ---- open SEG-Y volumes ----
            sources = [segyio.open(p, "r", ignore_geometry=True) for p in segy_list]
            try:
                if self._input_is_multiline:
                    tfs_line = self.time_first_sample[line_idx]  # type: ignore[index]
                else:
                    tfs_line = self.time_first_sample
                samples_ref = _build_samples_axis(
                    sources[0], tfs_line[0], self.dt, self.ns
                )
                nz = samples_ref.size
                ntr = sources[0].tracecount

                samples_per_volume = [
                    _build_samples_axis(src, tfs_line[v], self.dt, self.ns)
                    for v, src in enumerate(sources)
                ]
                index_maps = [
                    _build_index_map(samples_ref, samples_v)
                    for samples_v in samples_per_volume
                ]

                # ---- build top/base sample indices per aligned trace ----
                top_idx_map: Dict[int, int] = {}
                base_idx_map: Dict[int, int] = {}
                for ref_idx, t_top, t_base in zip(aligned_ref_indices, top_t, base_t):
                    if not np.isfinite(t_top) or not np.isfinite(t_base):
                        continue
                    iz_top = _time_to_index(samples_ref, float(t_top))
                    iz_base = _time_to_index(samples_ref, float(t_base))
                    if iz_top > iz_base:
                        iz_top, iz_base = iz_base, iz_top
                    top_idx_map[int(ref_idx)] = int(iz_top)
                    base_idx_map[int(ref_idx)] = int(iz_base)

                # ---- build training samples (accumulate across lines) ----
                p = self.sample_percent / 100.0

                for j, ref_idx in enumerate(aligned_ref_indices):
                    ref_idx = int(ref_idx)
                    if ref_idx not in top_idx_map:
                        continue

                    iz_top = top_idx_map[ref_idx]
                    iz_base = base_idx_map[ref_idx]

                    if p >= 1.0:
                        selected = list(range(iz_top, iz_base + 1))
                    else:
                        selected = [
                            iz for iz in range(iz_top, iz_base + 1)
                            if rng.random_sample() < p
                        ]

                    if not selected:
                        continue

                    traces = []
                    for v, src in enumerate(sources):
                        ti = aligned_trace_indices_per_volume[v][j]
                        traces.append(np.asarray(src.trace[ti], dtype=np.float32))

                    for iz in selected:
                        feat = [tr[index_maps[v][iz]] for v, tr in enumerate(traces)]
                        X_train.append(feat)
            finally:
                for src in sources:
                    src.close()

        if not X_train:
            raise ValueError("No training samples collected. Increase sample_percent.")

        X_train = np.asarray(X_train, dtype=float)
        invalid = np.any(np.isnan(X_train), axis=1)
        X_train = X_train[~invalid]
        if X_train.size == 0:
            raise ValueError("All training samples are NaN after cleaning.")

        # ---- fit single clusterer ----
        clusterer = Clusterer(model_type=self.model_type)
        clusterer.fit(X_train, parameters=self.cluster_params, scale_x=True)
        if self.model_output:
            clusterer.save(self.model_output)

        # ---- second pass: write output SEG-Y per line ----
        for line_idx in range(self._n_lines):
            segy_list, output_path = _get_line_inputs(line_idx)

            if self._input_is_multiline:
                hb_line = self.header_bytes[line_idx]  # type: ignore[index]
            else:
                hb_line = self.header_bytes
            headers_list = [
                self._read_headers(p, hb) for p, hb in zip(segy_list, hb_line)
            ]
            ref_headers = headers_list[0]

            mappings: List[Dict[Tuple[int, int], int]] = []
            for h in headers_list:
                if "INLINE" in h and "XLINE" in h:
                    keys = [
                        (int(il), int(xl))
                        for il, xl in zip(h["INLINE"], h["XLINE"])
                    ]
                elif "CDP" in h:
                    keys = [
                        (int(cdp), 0)
                        for cdp in h["CDP"]
                    ]
                else:
                    raise ValueError("Header missing INLINE/XLINE or CDP.")
                mapping = {k: i for i, k in enumerate(keys)}
                mappings.append(mapping)

            intersect_keys = set(mappings[0].keys())
            for m in mappings[1:]:
                intersect_keys &= set(m.keys())
            if not intersect_keys:
                raise ValueError("No intersecting X/Y keys across attributes.")

            if "INLINE" in ref_headers and "XLINE" in ref_headers:
                ref_keys = [
                    (int(il), int(xl))
                    for il, xl in zip(ref_headers["INLINE"], ref_headers["XLINE"])
                ]
            else:
                ref_keys = [
                    (int(cdp), 0)
                    for cdp in ref_headers["CDP"]
                ]
            aligned_keys = [k for k in ref_keys if k in intersect_keys]
            aligned_ref_indices = [mappings[0][k] for k in aligned_keys]
            aligned_trace_indices_per_volume = [
                [m[k] for k in aligned_keys] for m in mappings
            ]
            ref_idx_to_j = {int(ref_idx): j for j, ref_idx in enumerate(aligned_ref_indices)}

            ref_x = ref_headers["X"][aligned_ref_indices]
            ref_y = ref_headers["Y"][aligned_ref_indices]
            top_t = _nearest_time(top_tree, top_time, ref_x, ref_y)
            base_t = _nearest_time(base_tree, base_time, ref_x, ref_y)

            sources = [segyio.open(p, "r", ignore_geometry=True) for p in segy_list]
            try:
                if self._input_is_multiline:
                    tfs_line = self.time_first_sample[line_idx]  # type: ignore[index]
                else:
                    tfs_line = self.time_first_sample
                samples_ref = _build_samples_axis(
                    sources[0], tfs_line[0], self.dt, self.ns
                )
                nz = samples_ref.size
                ntr = sources[0].tracecount

                samples_per_volume = [
                    _build_samples_axis(src, tfs_line[v], self.dt, self.ns)
                    for v, src in enumerate(sources)
                ]
                index_maps = [
                    _build_index_map(samples_ref, samples_v)
                    for samples_v in samples_per_volume
                ]

                top_idx_map: Dict[int, int] = {}
                base_idx_map: Dict[int, int] = {}
                for ref_idx, t_top, t_base in zip(aligned_ref_indices, top_t, base_t):
                    if not np.isfinite(t_top) or not np.isfinite(t_base):
                        continue
                    iz_top = _time_to_index(samples_ref, float(t_top))
                    iz_base = _time_to_index(samples_ref, float(t_base))
                    if iz_top > iz_base:
                        iz_top, iz_base = iz_base, iz_top
                    top_idx_map[int(ref_idx)] = int(iz_top)
                    base_idx_map[int(ref_idx)] = int(iz_base)

                # copy template from first attribute of this line
                shutil.copyfile(segy_list[0], output_path)

                with segyio.open(output_path, "r+", ignore_geometry=True) as dst:
                    for i in range(ntr):
                        out = np.full(nz, self.null_value, dtype=np.float32)

                        if i in top_idx_map:
                            iz_top = top_idx_map[i]
                            iz_base = base_idx_map[i]

                            j = ref_idx_to_j.get(int(i), -1)
                            if j >= 0:
                                traces = []
                                for v, src in enumerate(sources):
                                    ti = aligned_trace_indices_per_volume[v][j]
                                    traces.append(np.asarray(src.trace[ti], dtype=np.float32))

                                idx_ref = list(range(iz_top, iz_base + 1))
                                cols = []
                                for v, tr in enumerate(traces):
                                    idx_v = index_maps[v][idx_ref]
                                    cols.append(tr[idx_v])
                                feats = np.column_stack(cols)
                                labels = clusterer.predict(feats, null_value=self.null_value)

                                if labels.ndim == 1:
                                    for k, iz in enumerate(range(iz_top, iz_base + 1)):
                                        lab = labels[k]
                                        if np.isnan(lab):
                                            out[iz] = self.null_value
                                        else:
                                            out[iz] = float(lab + self.label_offset)

                        dst.trace[i] = out
            finally:
                for src in sources:
                    src.close()
