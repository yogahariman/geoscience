import numpy as np
from .base.trace import TraceAttribute
from .instantaneous_amplitude import InstantaneousAmplitude

class DerivativeInstantaneousAmplitude(TraceAttribute):
    def process_trace(self, trace):
        amp = InstantaneousAmplitude.process_trace(self, trace)
        out = np.copy(trace)
        out[:-1] = np.diff(amp) / 4
        out[-1] = out[-2]
        return out