from __future__ import annotations

import shutil
from typing import Dict, List, Tuple, Union

import numpy as np
import segyio

from geosc.ml import Clusterer
from .base import SeismicMLBase


class SeismicClusterer(SeismicMLBase):
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
        t0: Union[float, List, None] = None,
        dt: Union[float, List, None] = None,
        ns: Union[int, List, None] = None,
        label_offset: int = 0,
        model_output: str | None = None,
    ) -> None:
        if sample_percent <= 0 or sample_percent > 100:
            raise ValueError("sample_percent must be in (0, 100].")

        super().__init__(
            input_segy_list=input_segy_list,
            output_segy=output_segy,
            horizon_top=horizon_top,
            horizon_base=horizon_base,
            header_bytes=header_bytes,
            null_value=null_value,
            t0=t0,
            dt=dt,
            ns=ns,
        )

        self.sample_percent = sample_percent
        params = dict(cluster_params or {})
        if "scale_x" in params:
            self.scale_x = bool(params.pop("scale_x"))
        elif "use_scaler" in params:
            self.scale_x = bool(params.pop("use_scaler"))
        else:
            self.scale_x = True
        self.cluster_params = params
        self.seed = seed
        self.model_type = model_type
        self.label_offset = int(label_offset)
        self.model_output = model_output

    def run(self) -> None:
        rng = np.random.RandomState(self.seed)

        # ---- load horizon (matrix Nx3) ----
        top_tree, top_time = self._build_horizon_tree(self.horizon_top)
        base_tree, base_time = self._build_horizon_tree(self.horizon_base)

        X_train: List[List[float]] = []

        for line_idx in range(self._n_lines):
            segy_list, output_path = self._get_line_inputs(line_idx)
            _ = output_path

            # ---- read headers for each attribute ----
            if self._input_is_multiline:
                hb_line = self.header_bytes[line_idx]  # type: ignore[index]
            else:
                hb_line = self.header_bytes
            headers_list = [
                self._read_headers(p, hb) for p, hb in zip(segy_list, hb_line)
            ]
            (
                aligned_ref_indices,
                aligned_trace_indices_per_volume,
                ref_idx_to_j,
                ref_x,
                ref_y,
            ) = self._build_alignment(headers_list)

            top_t = self._nearest_time(top_tree, top_time, ref_x, ref_y)
            base_t = self._nearest_time(base_tree, base_time, ref_x, ref_y)

            # ---- open SEG-Y volumes ----
            sources = [segyio.open(p, "r", ignore_geometry=True) for p in segy_list]
            try:
                if self._input_is_multiline:
                    tfs_line = self.t0[line_idx]  # type: ignore[index]
                    dt_line = self.dt[line_idx]  # type: ignore[index]
                    ns_line = self.ns[line_idx]  # type: ignore[index]
                else:
                    tfs_line = self.t0
                    dt_line = self.dt
                    ns_line = self.ns
                samples_ref = self._build_samples_axis(
                    sources[0], tfs_line[0], dt_line[0], ns_line[0]
                )

                samples_per_volume = [
                    self._build_samples_axis(src, tfs_line[v], dt_line[v], ns_line[v])
                    for v, src in enumerate(sources)
                ]
                index_maps = [
                    self._build_index_map(samples_ref, samples_v)
                    for samples_v in samples_per_volume
                ]

                top_idx_map, base_idx_map = self._build_interval_index_maps(
                    aligned_ref_indices=aligned_ref_indices,
                    top_t=top_t,
                    base_t=base_t,
                    samples_ref=samples_ref,
                )

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
        clusterer.fit(
            X_train,
            parameters=self.cluster_params,
            scale_x=self.scale_x,
        )
        if self.model_output:
            clusterer.save(self.model_output)

        # ---- second pass: write output SEG-Y per line ----
        for line_idx in range(self._n_lines):
            segy_list, output_path = self._get_line_inputs(line_idx)

            if self._input_is_multiline:
                hb_line = self.header_bytes[line_idx]  # type: ignore[index]
            else:
                hb_line = self.header_bytes
            headers_list = [
                self._read_headers(p, hb) for p, hb in zip(segy_list, hb_line)
            ]
            (
                aligned_ref_indices,
                aligned_trace_indices_per_volume,
                ref_idx_to_j,
                ref_x,
                ref_y,
            ) = self._build_alignment(headers_list)
            top_t = self._nearest_time(top_tree, top_time, ref_x, ref_y)
            base_t = self._nearest_time(base_tree, base_time, ref_x, ref_y)

            sources = [segyio.open(p, "r", ignore_geometry=True) for p in segy_list]
            try:
                if self._input_is_multiline:
                    tfs_line = self.t0[line_idx]  # type: ignore[index]
                    dt_line = self.dt[line_idx]  # type: ignore[index]
                    ns_line = self.ns[line_idx]  # type: ignore[index]
                else:
                    tfs_line = self.t0
                    dt_line = self.dt
                    ns_line = self.ns
                samples_ref = self._build_samples_axis(
                    sources[0], tfs_line[0], dt_line[0], ns_line[0]
                )
                nz = samples_ref.size
                ntr = sources[0].tracecount

                samples_per_volume = [
                    self._build_samples_axis(src, tfs_line[v], dt_line[v], ns_line[v])
                    for v, src in enumerate(sources)
                ]
                index_maps = [
                    self._build_index_map(samples_ref, samples_v)
                    for samples_v in samples_per_volume
                ]

                top_idx_map, base_idx_map = self._build_interval_index_maps(
                    aligned_ref_indices=aligned_ref_indices,
                    top_t=top_t,
                    base_t=base_t,
                    samples_ref=samples_ref,
                )

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
