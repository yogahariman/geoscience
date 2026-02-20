from __future__ import annotations

from typing import Dict, Tuple

from .crop import crop_segy_3d
from .resample import resample_segy_3d


class SeismicEditor:
    """
    Utility container for seismic editing operations.
    """

    @staticmethod
    def crop_3d(
        input_segy: str,
        output_segy: str,
        inline_range: Tuple[int, int],
        xline_range: Tuple[int, int],
        time_range: Tuple[float, float] | None = None,
        t0: float | None = None,
        header_bytes: Dict[str, Tuple[int, str]] | None = None,
    ) -> None:
        crop_segy_3d(
            input_segy=input_segy,
            output_segy=output_segy,
            inline_range=inline_range,
            xline_range=xline_range,
            time_range=time_range,
            t0=t0,
            header_bytes=header_bytes,
        )

    @staticmethod
    def resample_3d(
        input_segy: str,
        output_segy: str,
        inline_step: int = 2,
        xline_step: int = 2,
        header_bytes: Dict[str, Tuple[int, str]] | None = None,
    ) -> None:
        resample_segy_3d(
            input_segy=input_segy,
            output_segy=output_segy,
            inline_step=inline_step,
            xline_step=xline_step,
            header_bytes=header_bytes,
        )
