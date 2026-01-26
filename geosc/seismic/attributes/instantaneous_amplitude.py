import numpy as np
from scipy.signal import hilbert
from .base.trace import TraceAttribute

class InstantaneousAmplitude(TraceAttribute):
    def process_trace(self, trace):
        analytic = hilbert(trace)
        return np.sqrt(trace**2 + np.imag(analytic)**2)
