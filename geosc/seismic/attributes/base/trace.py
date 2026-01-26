import segyio
import numpy as np


class TraceAttribute:
    """
    Base class for trace-based seismic attributes.
    Each trace is processed independently.
    """

    def __init__(self, input_segy, output_segy):
        self.input_segy = input_segy
        self.output_segy = output_segy

    # ==========================================================
    # TO BE IMPLEMENTED BY CHILD CLASS
    # ==========================================================
    def process_trace(self, trace: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self):
        with segyio.open(self.input_segy, "r", ignore_geometry=True) as src:
            nz = len(src.samples)
            ntr = src.tracecount

            spec = segyio.spec()
            spec.samples = src.samples
            spec.format = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE
            spec.tracecount = ntr

            with segyio.create(self.output_segy, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin

                for i in range(ntr):
                    trace = src.trace[i].astype(np.float32)

                    out = self.process_trace(trace)

                    if out.shape[0] != nz:
                        raise ValueError("Output trace length mismatch")

                    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

                    dst.trace[i] = out
                    dst.header[i] = src.header[i]
