from __future__ import annotations

import shutil
from typing import Optional

import numpy as np
import segyio


class SeismicHydrostatic:
    """
    Compute hydrostatic result from depth TVDSS seismic volume.

    Input:
    - depth TVDSS volume in meters (SEG-Y)

    Output unit (`output_unit`):
    - ``"psi"`` : hydrostatic pressure in psi
    - ``"sg"``  : hydrostatic equivalent specific gravity
    """

    PA_TO_PSI = 0.00014503773773

    def __init__(
        self,
        input_segy: str,
        output_segy: str,
        output_unit: str = "psi",
        density: float = 1000.0,
        gravity: float = 9.78033,
        null_value: Optional[float] = -999.25,
    ) -> None:
        unit = output_unit.strip().lower()
        if unit not in {"psi", "sg"}:
            raise ValueError("output_unit must be 'psi' or 'sg'.")
        if density <= 0:
            raise ValueError("density must be > 0.")
        if gravity <= 0:
            raise ValueError("gravity must be > 0.")

        self.input_segy = input_segy
        self.output_segy = output_segy
        self.output_unit = unit
        self.density = float(density)
        self.gravity = float(gravity)
        self.null_value = None if null_value is None else float(null_value)

    def _convert_depth(self, depth_m: np.ndarray) -> np.ndarray:
        # TVDSS commonly stored negative below sea level; use magnitude for pressure.
        z = np.abs(depth_m.astype(float))
        pressure_pa = self.density * self.gravity * z

        if self.output_unit == "psi":
            return pressure_pa * self.PA_TO_PSI

        # SG equivalent from hydrostatic relation.
        out = np.zeros_like(z, dtype=float)
        nonzero = z > 0.0
        out[nonzero] = pressure_pa[nonzero] / (self.gravity * 1000.0 * z[nonzero])
        return out

    def run(self) -> None:
        shutil.copyfile(self.input_segy, self.output_segy)

        with segyio.open(self.input_segy, "r", ignore_geometry=True) as src, segyio.open(
            self.output_segy, "r+", ignore_geometry=True
        ) as dst:
            for i in range(src.tracecount):
                trace = np.asarray(src.trace[i], dtype=float)
                if self.null_value is None:
                    out = np.full(trace.shape, np.nan, dtype=np.float32)
                    valid = np.isfinite(trace)
                else:
                    out = np.full(trace.shape, self.null_value, dtype=np.float32)
                    valid = np.isfinite(trace) & (trace != self.null_value)
                if np.any(valid):
                    out_vals = self._convert_depth(trace[valid]).astype(np.float32)
                    out[valid] = out_vals

                dst.trace[i] = out
