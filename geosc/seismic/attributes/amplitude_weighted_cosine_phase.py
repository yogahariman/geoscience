import numpy as np
from .base.trace import TraceAttribute
from .instantaneous_amplitude import InstantaneousAmplitude
from .cosine_instantaneous_phase import CosineInstantaneousPhase

class AmplitudeWeightedCosinePhase(TraceAttribute):
    def process_trace(self, trace):
        amp = InstantaneousAmplitude.process_trace(self, trace)
        cos_phase = CosineInstantaneousPhase.process_trace(self, trace)
        return amp * cos_phase
