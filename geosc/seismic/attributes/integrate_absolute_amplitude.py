import numpy as np
from .base.trace import TraceAttribute
from .instantaneous_amplitude import InstantaneousAmplitude

class IntegrateAbsoluteAmplitude(TraceAttribute):
    def __init__(self, input_segy, output_segy, k=56):
        super().__init__(input_segy, output_segy)
        self.k = k

    def process_trace(self, trace):
        e = InstantaneousAmplitude.process_trace(self, trace)
        k = self.k

        pad = np.zeros(len(e) + 2 * k)
        pad[k:k+len(e)] = e

        es = np.zeros_like(e)
        half = k // 2

        for i in range(len(e)):
            es[i] = np.sum(pad[i + k - half : i + k + half])

        es /= (2 * k + 1)
        return np.abs(e - es) * 2 / 100
