import numpy as np
from .base.trace import TraceAttribute
from .instantaneous_amplitude import InstantaneousAmplitude
from .instantaneous_frequency import InstantaneousFrequency

class AmplitudeWeightedFrequency(TraceAttribute):
    def process_trace(self, trace):
        amp = InstantaneousAmplitude.process_trace(self, trace)
        freq = InstantaneousFrequency.process_trace(self, trace)
        return amp * freq
