import numpy as np
from .base.trace import TraceAttribute


class SecondDerivative(TraceAttribute):
    def process_trace(self, trace):
        out = np.copy(trace)
        out[:-2] = np.diff(trace, 2)
        out[-2] = out[-3]
        out[-1] = out[-3]
        return out