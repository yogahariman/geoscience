import numpy as np
from .base.trace import TraceAttribute
from .instantaneous_phase import InstantaneousPhase

class CosineInstantaneousPhase(TraceAttribute):
    def process_trace(self, trace):
        phase = InstantaneousPhase.process_trace(self, trace)
        return np.cos(np.radians(phase))

