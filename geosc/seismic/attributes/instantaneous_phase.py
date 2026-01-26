import numpy as np
from scipy.signal import hilbert
from .base.trace import TraceAttribute

class InstantaneousPhase(TraceAttribute):
    def process_trace(self, trace):
        analytic = hilbert(trace)
        return np.degrees(np.arctan2(np.imag(analytic), trace))
