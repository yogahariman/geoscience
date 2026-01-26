import numpy as np
from scipy.signal import hilbert
from .base.trace import TraceAttribute


class InstantaneousFrequency(TraceAttribute):
    def process_trace(self, trace):
        phase = np.unwrap(np.angle(hilbert(trace)))
        freq = np.abs(np.diff(phase) * 180 / np.pi)

        out = np.empty_like(trace)
        out[:-1] = freq
        out[-1] = freq[-1]
        return out
