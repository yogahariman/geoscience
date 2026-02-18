from __future__ import annotations

import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import segyio

from geosc.seismic.segy import get_segy_trace_headers


class SeismicOverburden:
    """
    Compute overburden from TVDSS and density seismic volumes.

    Input:
    - TVDSS volume in meter
    - Density volume in kg/m^3

    Output unit (`output_unit`):
    - ``"psi"`` : overburden pressure in psi
    - ``"sg"``  : equivalent mud gradient (specific gravity)
    """

    PA_TO_PSI = 0.00014503773773

    def __init__(
        self,
        tvdss_segy: str,
        density_segy: str,
        output_segy: str,
        tvdss_header_bytes: Dict[str, Tuple[int, str]],
        density_header_bytes: Dict[str, Tuple[int, str]],
        output_unit: str = "psi",
        density_unit: str = "g/cc",
        gravity: float = 9.78033,
        null_value: Optional[float] = -999.25,
    ) -> None:
        unit = output_unit.strip().lower()
        if unit not in {"psi", "sg"}:
            raise ValueError("output_unit must be 'psi' or 'sg'.")
        density_unit_norm = density_unit.strip().lower().replace(" ", "")
        if density_unit_norm in {"g/cc", "gcc", "g/cm3", "g/cm^3"}:
            density_unit_norm = "g/cc"
        elif density_unit_norm in {"kg/m3", "kg/m^3"}:
            density_unit_norm = "kg/m3"
        else:
            raise ValueError("density_unit must be 'g/cc' or 'kg/m3'.")
        if gravity <= 0:
            raise ValueError("gravity must be > 0.")

        self.tvdss_segy = tvdss_segy
        self.density_segy = density_segy
        self.output_segy = output_segy
        self.output_unit = unit
        self.density_unit = density_unit_norm
        self.gravity = float(gravity)
        self.null_value = None if null_value is None else float(null_value)
        self.tvdss_header_bytes = {k.upper(): v for k, v in tvdss_header_bytes.items()}
        self.density_header_bytes = {k.upper(): v for k, v in density_header_bytes.items()}

        self._validate_header_bytes(self.tvdss_header_bytes, "tvdss_header_bytes")
        self._validate_header_bytes(self.density_header_bytes, "density_header_bytes")

    @staticmethod
    def _validate_header_bytes(header_bytes: Dict[str, Tuple[int, str]], name: str) -> None:
        keys = set(header_bytes.keys())
        if not ({"INLINE", "XLINE"} <= keys or "CDP" in keys):
            raise ValueError(f"{name} must include INLINE+XLINE (3D) or CDP (2D).")

    @staticmethod
    def _trace_keys(headers: Dict[str, np.ndarray]) -> list[tuple[int, int]]:
        if "INLINE" in headers and "XLINE" in headers:
            return [(int(il), int(xl)) for il, xl in zip(headers["INLINE"], headers["XLINE"])]
        if "CDP" in headers:
            return [(int(cdp), 0) for cdp in headers["CDP"]]
        raise ValueError("Headers must contain INLINE/XLINE or CDP.")

    @staticmethod
    def _read_headers(path: str, hb: Dict[str, Tuple[int, str]]) -> Dict[str, np.ndarray]:
        header_map: Dict[str, Tuple[int, str]] = {}
        if "INLINE" in hb and "XLINE" in hb:
            header_map["INLINE"] = hb["INLINE"]
            header_map["XLINE"] = hb["XLINE"]
        if "CDP" in hb:
            header_map["CDP"] = hb["CDP"]
        return get_segy_trace_headers(path, header_map)

    @staticmethod
    def _resample_to_tvdss_length(arr: np.ndarray, n_out: int) -> np.ndarray:
        if arr.size == n_out:
            return arr
        if arr.size == 1:
            return np.full(n_out, float(arr[0]), dtype=float)
        x_old = np.linspace(0.0, 1.0, arr.size)
        x_new = np.linspace(0.0, 1.0, n_out)
        return np.interp(x_new, x_old, arr)

    def _compute_trace(self, tvdss_trace: np.ndarray, density_trace: np.ndarray) -> np.ndarray:
        z = np.asarray(tvdss_trace, dtype=float)
        rho = np.asarray(density_trace, dtype=float)
        rho = self._resample_to_tvdss_length(rho, z.size)
        if self.density_unit == "g/cc":
            rho = rho * 1000.0  # convert g/cc to kg/m3

        if self.null_value is None:
            valid = np.isfinite(z) & np.isfinite(rho)
            out = np.full(z.shape, np.nan, dtype=np.float32)
        else:
            valid = (
                np.isfinite(z)
                & np.isfinite(rho)
                & (z != self.null_value)
                & (rho != self.null_value)
            )
            out = np.full(z.shape, self.null_value, dtype=np.float32)

        if not np.any(valid):
            return out

        z_abs = np.abs(z[valid])
        z_mono = np.maximum.accumulate(z_abs)
        dz = np.diff(np.r_[0.0, z_mono])
        pressure_pa = np.cumsum(rho[valid] * self.gravity * dz)

        if self.output_unit == "psi":
            out_vals = pressure_pa * self.PA_TO_PSI
        else:
            out_vals = np.zeros_like(pressure_pa, dtype=float)
            nonzero = z_mono > 0.0
            out_vals[nonzero] = pressure_pa[nonzero] / (self.gravity * 1000.0 * z_mono[nonzero])

        out[valid] = out_vals.astype(np.float32)
        return out

    def run(self) -> None:
        tvd_headers = self._read_headers(self.tvdss_segy, self.tvdss_header_bytes)
        den_headers = self._read_headers(self.density_segy, self.density_header_bytes)

        tvd_keys = self._trace_keys(tvd_headers)
        den_keys = self._trace_keys(den_headers)
        den_map = {k: i for i, k in enumerate(den_keys)}

        shutil.copyfile(self.tvdss_segy, self.output_segy)

        with segyio.open(self.tvdss_segy, "r", ignore_geometry=True) as tvd_src, segyio.open(
            self.density_segy, "r", ignore_geometry=True
        ) as den_src, segyio.open(self.output_segy, "r+", ignore_geometry=True) as dst:
            for i, key in enumerate(tvd_keys):
                j = den_map.get(key)

                tvd_trace = np.asarray(tvd_src.trace[i], dtype=float)
                if j is None:
                    if self.null_value is None:
                        dst.trace[i] = np.full(tvd_trace.shape, np.nan, dtype=np.float32)
                    else:
                        dst.trace[i] = np.full(tvd_trace.shape, self.null_value, dtype=np.float32)
                    continue

                den_trace = np.asarray(den_src.trace[j], dtype=float)
                dst.trace[i] = self._compute_trace(tvd_trace, den_trace)
