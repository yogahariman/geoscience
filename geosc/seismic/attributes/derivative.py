import numpy as np
from .base.trace import TraceAttribute

class Derivative(TraceAttribute):
    def process_trace(self, trace):
        out = np.copy(trace)
        out[:-1] = np.diff(trace)
        out[-1] = out[-2]
        return out
