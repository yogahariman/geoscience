import numpy as np
from scipy.signal import hilbert
from .base.trace import TraceAttribute

class Quadrature(TraceAttribute):
    def process_trace(self, trace):
        return np.imag(hilbert(trace))