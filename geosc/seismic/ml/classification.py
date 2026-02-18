from __future__ import annotations

import shutil
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import segyio

from geosc.ml import Classifier, DataCleaner
from .base import SeismicMLBase


def _normalize_labels(
    labels: np.ndarray,
    null_value: float,
) -> np.ndarray:
    out = np.full(labels.shape[0], null_value, dtype=np.float32)
    for i, raw in enumerate(labels):
        try:
            value = float(raw)
        except Exception as exc:
            raise ValueError(
                "Predicted labels must be numeric to write SEG-Y. "
                "Use numeric classes in geosc.ml.classification training."
            ) from exc

        if np.isnan(value):
            out[i] = null_value
        else:
            out[i] = float(value)
    return out


class SeismicClassifier(SeismicMLBase):
    """
    Seismic classification within horizon interval, with multi-volume inputs.

    - Input volumes are aligned by X/Y headers.
    - Horizon top/base are provided as Nx3 [x,y,z/time] and matched by nearest x,y.
    - Model is loaded from geosc.ml.Classifier output (predict only).
    - Output can be label volume or probability volume for one class.
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
        output_mode: str = "labels",
        probability_class: Any = None,
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
        self.model_path = model_path
        self.output_mode = str(output_mode).lower().strip()
        self.probability_class = probability_class
        if self.output_mode not in {"labels", "prob_class", "max_prob"}:
            raise ValueError("output_mode must be 'labels', 'prob_class', or 'max_prob'.")

    def run(self) -> None:
        top_tree, top_time = self._build_horizon_tree(self.horizon_top)
        base_tree, base_time = self._build_horizon_tree(self.horizon_base)

        predictor = Classifier.load(self.model_path)
        cleaner = DataCleaner(null_value=self.null_value)
        prob_col: int | None = None
        if self.output_mode == "prob_class":
            if self.probability_class is None:
                raise ValueError(
                    "probability_class is required when output_mode='prob_class'."
                )
            classes = list(getattr(predictor, "classes_", []) or [])
            if not classes:
                raise ValueError(
                    "Model classes_ not found. Re-save model from geosc.ml.Classifier."
                )
            try:
                prob_col = classes.index(self.probability_class)
            except ValueError as exc:
                raise ValueError(
                    f"Class {self.probability_class!r} not found in model classes: {classes!r}"
                ) from exc

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

                                labels, probs = predictor.predict(feats, null_value=self.null_value)
                                if self.output_mode == "labels":
                                    out_vals = _normalize_labels(
                                        labels=np.asarray(labels),
                                        null_value=self.null_value,
                                    )
                                elif self.output_mode == "prob_class":
                                    if probs is None:
                                        raise ValueError(
                                            "Model does not provide probabilities. "
                                            "Use model with predict_proba/decision_function support."
                                        )
                                    if prob_col is None or prob_col >= probs.shape[1]:
                                        raise ValueError(
                                            "Invalid probability class column index for model output."
                                        )
                                    out_vals = np.full(probs.shape[0], self.null_value, dtype=np.float32)
                                    valid = np.isfinite(probs[:, prob_col])
                                    out_vals[valid] = probs[valid, prob_col].astype(np.float32)
                                else:
                                    if probs is None:
                                        raise ValueError(
                                            "Model does not provide probabilities. "
                                            "Use model with predict_proba/decision_function support."
                                        )
                                    probs_f = np.asarray(probs, dtype=float)
                                    probs_f[probs_f == self.null_value] = np.nan
                                    out_vals = np.full(probs_f.shape[0], self.null_value, dtype=np.float32)
                                    valid = np.any(np.isfinite(probs_f), axis=1)
                                    if np.any(valid):
                                        out_vals[valid] = np.nanmax(probs_f[valid], axis=1).astype(np.float32)

                                out[np.asarray(idx_ref, dtype=int)] = out_vals

                        dst.trace[i] = out
            finally:
                for src in sources:
                    src.close()
