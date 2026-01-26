import numpy as np
from .base.trace import TraceAttribute
from .instantaneous_amplitude import InstantaneousAmplitude

class SecondDerivativeInstantaneousAmplitude(TraceAttribute):
    def process_trace(self, trace):
        amp = InstantaneousAmplitude.process_trace(self, trace)
        out = np.copy(trace)
        out[:-2] = np.diff(amp, 2) / 16
        out[-2] = out[-3]
        out[-1] = out[-3]
        return out
