# Geoscience Seismic Attributes

## Installation
pip install -e .

## Quick Example
```python
from seismic.attributes import Coherence3D

Coherence3D(
    "input.sgy",
    "coh.sgy",
    window=(3,1,1),
    inline_byte=121,
    xline_byte=125
).run()
