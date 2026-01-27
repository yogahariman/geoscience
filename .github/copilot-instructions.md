# Copilot Instructions for Geoscience Toolbox

## Project Overview
**geoscience** is a Python library for seismic data processing and geoanalysis. It provides:
- **Seismic attributes**: 20+ signal processing algorithms (coherence, instantaneous phase, amplitude-weighted derivatives, etc.)
- **SEG-Y I/O**: Binary seismic data format reading/writing with header management
- **Map/Gridding**: Spatial interpolation (IDW, kriging via scikit-gstat)
- **Well data tools**: Stub for future well log integration
- **Plotting**: 2D/3D seismic visualization

The project is designed as a pip-installable package (`pip install -e .`) with modular architecture separating low-level I/O from high-level algorithms.

## Architecture & Key Patterns

### 1. Attribute Processing Pattern (Core Design)
All seismic attributes follow two inheritance hierarchies:

**TraceAttribute** (independent processing):
```python
class MyTraceAttribute(TraceAttribute):
    def process_trace(self, trace: np.ndarray) -> np.ndarray:
        # Process single trace, e.g., FFT, derivatives, integration
        return processed_trace
    
    def run(self):  # Inherited - handles I/O, NaN cleanup
```
Example: `InstantaneousAmplitude`, `Derivative`

**WindowAttribute** (spatially-local processing):
```python
class MyWindowAttribute(WindowAttribute):
    def process_window(self, window: np.ndarray) -> float:
        # Process 2D/3D window, return single scalar
        return metric
    
    def run(self):  # Inherited - tile data, handle boundaries
```
Examples: `Coherence2D`, `Coherence3D`, `Semblance2D`

**Key design decisions:**
- Base classes handle SEG-Y I/O and memory management (see `load_to_ram` option)
- Child classes ONLY implement domain logic
- No header parsing in attributes; use explicit inline/xline arrays for 3D
- Output is always float32; NaN→0 substitution automatic
- Window attributes clip at boundaries (no padding)

### 2. Project Structure
```
geosc/
  seismic/
    segy/               ← Low-level SEG-Y format (headers, endianness)
    attributes/
      base/             ← TraceAttribute, WindowAttribute base classes
      *.py              ← 20+ algorithm implementations
    plotting/           ← SeismicPlot2D, SeismicPlot3D (matplotlib-based)
  map/
    gridding/           ← IDW & kriging interpolation
  well/                 ← Placeholder for future development
scripts/                ← CLI scripts and test examples
```

### 3. Dependencies
**Core:** numpy, scipy, segyio (SEG-Y format)  
**Optional:** scikit-gstat (kriging in map/gridding)  
**Visualization:** matplotlib (in plotting/)

### 4. Testing & Scripts
- `scripts/test_attribute_*.py`: Examples showing attribute usage
- Pattern: Load input SEG-Y → instantiate attribute class → call `.run()` → output SEG-Y
- Test scripts also demonstrate plotting: `SeismicPlot2D(segy_path).plot(plot_type="amplitude")`

## Conventions & Developer Workflows

### Adding a New Seismic Attribute
1. Create file `geosc/seismic/attributes/myattribute.py`
2. Inherit from `TraceAttribute` (1 value per trace) or `WindowAttribute` (1 value per window)
3. Implement `process_trace()` or `process_window()` logic
4. Add import + export in `geosc/seismic/attributes/__init__.py` under appropriate comment section
5. Create test script in `scripts/test_myattribute.py` following existing patterns
6. Base class `.run()` handles: SEG-Y I/O, iteration, NaN cleanup, output spec

### Running Attributes from Command Line
Attributes are instantiated and executed in scripts, not CLI-driven:
```bash
cd /Drive/D/geoscience
pip install -e .
python scripts/test_attribute_2d.py
```

### Critical Code Paths
- **SEG-Y I/O:** Base classes use `segyio` library; all I/O logic in `TraceAttribute.run()` and `WindowAttribute.run()`
- **Memory optimization:** `WindowAttribute` has `load_to_ram` flag for processing large 3D cubes
- **Numerical safety:** Automatic NaN→0, inf→0 substitution in base class `.run()`
- **Geometry handling (3D):** Explicit inline/xline arrays required; no byte position assumptions in attributes

### Naming Conventions
- Algorithm classes: PascalCase matching the geophysical concept (e.g., `InstantaneousPhase`, `AmplitudeWeightedCosinePhase`)
- Utility functions: snake_case (e.g., `idw_interpolate()`)
- Private methods in base classes prefixed with `_` (e.g., `_load_to_ram()`)
- Comments group code in base classes: `# ==========================================================`

## Common Integration Points

### 1. Adding Gridding Interpolation
File: `geosc/map/gridding/utils.py`
- `idw_interpolate(x, y, values, grid_x, grid_y, power=2)`: Inverse distance weighting
- `auto_variogram(x, y, values, model="spherical")`: Geostatistical kriging params
Pattern: Input XY point clouds + grid definition → 2D interpolated array

### 2. SEG-Y Header Access
File: `geosc/seismic/segy/trace_headers.py`
- Use `get_segy_trace_headers()` to batch-read all trace headers
- For byte position lookups: Pass explicit positions (e.g., `inline_byte=121`) to attribute classes

### 3. Plotting Integration
File: `geosc/seismic/plotting/plot2d.py`
- `SeismicPlot2D(segy_path).plot(plot_type="amplitude")` for visualization
- Used in test scripts post-processing

## Before Contributing Code
- **Test locally:** `python scripts/test_*.py` to validate attribute output
- **Check base class handling:** Verify I/O and memory options work (e.g., 3D coherence with `load_to_ram=True`)
- **Numerical stability:** Window attributes should gracefully handle edge cases (small windows, NaN-heavy data)
- **Documentation:** Add docstring to attribute class + example in `README.md` if major new feature
