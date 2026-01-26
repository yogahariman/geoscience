import numpy as np
from .base.trace import TraceAttribute
from .instantaneous_amplitude import InstantaneousAmplitude
from .instantaneous_phase import InstantaneousPhase

class AmplitudeWeightedPhase(TraceAttribute):
    def process_trace(self, trace):
        amp = InstantaneousAmplitude.process_trace(self, trace)
        phase = InstantaneousPhase.process_trace(self, trace)
        return amp * phase
