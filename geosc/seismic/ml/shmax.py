from __future__ import annotations

import shutil
from typing import Dict, List, Tuple, Union

import numpy as np
import segyio

from geosc.ml import DataCleaner, ShmaxRegressor
from .base import SeismicMLBase


class SeismicShmaxPredictor(SeismicMLBase):
    """
    Seismic Shmax prediction within horizon interval (predict-only).

    Workflow follows seismic ML regression:
    - align input volumes by geometry key (INLINE/XLINE or CDP)
    - extract samples only between horizon top/base
    - predict with trained ``geosc.ml.ShmaxRegressor``

    Important:
    - Input attribute order in ``input_segy_list`` must match training feature order.
    - Typically first features are [hydrostatic, overburden, porepressure, shmin, ...].
    """

    def __init__(
        self,
        input_segy_list: Union[List[str], List[List[str]]],
        output_segy: Union[str, List[str]],
        horizon_top: np.ndarray,
        horizon_base: np.ndarray,
        header_bytes: List[Dict[str, Tuple[int, str]]],
        model_path: str,
        null_value: float = -999.25,
        t0: Union[float, List, None] = None,
        dt: Union[float, List, None] = None,
        ns: Union[int, List, None] = None,
    ) -> None:
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
        if self._n_attrs < 4:
            raise ValueError(
                "Shmax predictor requires at least 4 input attributes: "
                "[hydrostatic, overburden, porepressure, shmin, ...]."
            )
        self.model_path = model_path

    @staticmethod
    def load_model(model_path: str) -> ShmaxRegressor:
        return ShmaxRegressor.load(model_path)

    def run(self) -> None:
        top_tree, top_time = self._build_horizon_tree(self.horizon_top)
        base_tree, base_time = self._build_horizon_tree(self.horizon_base)

        predictor = self.load_model(self.model_path)
        cleaner = DataCleaner(null_value=self.null_value)

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
                                feats = cleaner.clean_data_prediction(feats)

                                preds = predictor.predict(feats, null_value=self.null_value)
                                out_vals = np.asarray(preds, dtype=np.float32)
                                invalid = ~np.isfinite(out_vals)
                                out_vals[invalid] = self.null_value
                                out[np.asarray(idx_ref, dtype=int)] = out_vals

                        dst.trace[i] = out
            finally:
                for src in sources:
                    src.close()
