import numpy as np
from .base.trace import TraceAttribute
from scipy.integrate import cumulative_trapezoid as cumtrapz

class IntegrateTraces(TraceAttribute):
    def process_trace(self, trace):
        return cumtrapz(trace, initial=trace[0])