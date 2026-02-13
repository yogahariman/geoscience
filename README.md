# Geoscience Toolbox

A comprehensive Python library for seismic data processing, geoanalysis, and machine learning-driven well log analysis.

## Features

### 🔄 Seismic Data Processing
- **20+ Seismic Attributes**: Signal processing algorithms including:
  - Instantaneous phase & amplitude
  - Coherence (2D/3D)
  - Amplitude-weighted derivatives
  - Semblance analysis
  - And more...

### 📁 SEG-Y Format Support
- Binary seismic data reading/writing
- Full header management
- Efficient I/O with optional RAM caching

### 🗺️ Spatial Gridding & Interpolation
- Inverse Distance Weighting (IDW)
- Kriging via scikit-gstat
- 2D/3D grid generation

### 🤖 Machine Learning for Well Logs
- **Classification**: Multiple model types (XGBoost, Random Forest, MLP, SVM, Naive Bayes)
- **Regression**: Continuous value prediction
- **Clustering**: Unsupervised learning (K-means, DBSCAN, GMM, Agglomerative)
- **Lithology Prediction**: Domain-specific classification
- Data cleaning & preprocessing utilities

### 🌊 Seismic ML
- **Seismic Clustering**: Horizon-constrained unsupervised clustering from multi-attribute SEG-Y volumes
- **Seismic Lithology Prediction**: Horizon-constrained prediction using trained `geosc.ml.Classifier` model
- **Seismic Classification**: Generic horizon-constrained classification using trained `geosc.ml.Classifier` model
- **Seismic Regression**: Horizon-constrained continuous prediction using trained `geosc.ml.Regressor` model
  - `output_mode="labels"`: predicted class labels
  - `output_mode="prob_class"`: probability of one selected class
  - `output_mode="max_prob"`: maximum class probability (model confidence)

### 📊 Visualization
- 2D seismic plot generation
- 3D seismic visualization
- Discrete lithology color plotting with custom class colors
- Well log cross-sections

## Installation

### Prerequisites
- Python 3.7+
- pip

### Quick Install

```bash
git clone https://github.com/yogahariman/geoscience.git
cd geoscience
pip install -e .
```

### Dependencies
- **Core**: numpy, scipy, segyio
- **Optional**: scikit-gstat (for kriging)
- **Visualization**: matplotlib

## Quick Start

### Processing Seismic Attributes

```python
from geosc.seismic.attributes import InstantaneousAmplitude

# Create and run attribute
attr = InstantaneousAmplitude(
    input_segy="data.sgy",
    output_segy="output_amplitude.sgy"
)
attr.run()
```

### 2D Coherence Analysis

```python
from geosc.seismic.attributes import Coherence2D

coherence = Coherence2D(
    input_segy="data.sgy",
    output_segy="coherence_output.sgy",
    window_inline=5,
    window_xline=5
)
coherence.run()
```

### Machine Learning: Well Log Classification

```python
from geosc.ml import DataCleaner, Classifier

# Prepare data
cleaner = DataCleaner(null_value=-999.25)
X_clean, y_clean = cleaner.clean_data_training(X, y)

# Train model
model = Classifier(model_type="mlp")
model.train(X_clean, y_clean, parameters={
    "hidden_layer_sizes": (128, 64),
    "max_iter": 1000
}, scale_x=True)

# Save model
model.save("lithology_model.pkl")

# Predict on new data
X_new = cleaner.clean_data_prediction(X_new)
predictions, probabilities = model.predict(X_new, null_value=-999.25)
```

### Seismic Visualization

```python
from geosc.seismic.plotting import SeismicPlot2D

plotter = SeismicPlot2D("data.sgy")
plotter.plot(plot_type="amplitude")
```

### Seismic Lithology Plot (Custom Colors)

```python
from geosc.seismic.plotting.lithology_utils import build_discrete_style

CLASS_COLORS = {
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "black",
}
```

## Project Structure

```
geosc/
├── plotting/                   # Generic plotting (tabular/crossplot/classification)
│   ├── classification_2d.py    # NaiveBayesGaussianContour2D
│
├── seismic/
│   ├── segy/                    # Low-level SEG-Y format I/O
│   ├── attributes/
│   │   ├── base/                # TraceAttribute, WindowAttribute base classes
│   │   ├── *.py                 # 20+ seismic attribute implementations
│   ├── plotting/                # SeismicPlot2D, SeismicPlot3D
│   ├── ml/                      # SeismicMLBase + SeismicClassifier/Regressor/Clusterer/LithologyPredictor
│
├── map/
│   ├── gridding/                # IDW & kriging interpolation
│
├── ml/                          # Machine Learning module
│   ├── cleaning.py              # DataCleaner: data preprocessing
│   ├── classification.py        # Classifier: supervised classification
│   ├── regression.py            # Regressor: supervised regression
│   ├── clustering.py            # Clusterer: unsupervised learning
│   ├── lithology.py             # LithologyPredictor: domain-specific
│
├── well/                        # Well log integration (future)
│
scripts/                         # CLI scripts & test examples
```

## Architecture Overview

### Attribute Processing Pattern

All seismic attributes follow a consistent pattern based on processing scope:

**TraceAttribute** - Process single traces independently:
```python
class MyTraceAttribute(TraceAttribute):
    def process_trace(self, trace: np.ndarray) -> np.ndarray:
        """Process single trace"""
        return processed_trace
```

**WindowAttribute** - Process 2D/3D windows for spatial metrics:
```python
class MyWindowAttribute(WindowAttribute):
    def process_window(self, window: np.ndarray) -> float:
        """Process window, return scalar metric"""
        return metric_value
```

### Key Design Principles
- Base classes handle I/O, iteration, and memory management
- Child classes implement domain logic only
- Automatic NaN→0 substitution
- Float32 output format
- 3D processing uses explicit inline/xline arrays

## ML Module Workflow

All ML classes provide unified APIs:

```python
# Data cleaning
cleaner = DataCleaner(null_value=-999.25)
X_train, y_train = cleaner.clean_data_training(X, y)

# Model training
model = Classifier(model_type="random_forest")  # or mlp, svm, xgboost, naive_bayes
model.train(X_train, y_train, parameters={...}, scale_x=True)
model.save("model.pkl")

# Prediction
X_new = cleaner.clean_data_prediction(X_new)
predictions, probabilities = model.predict(X_new, null_value=-999.25)
```

Supported clustering models: K-means, DBSCAN, Gaussian Mixture, Agglomerative

### Seismic Lithology Prediction Workflow

```python
from geosc.seismic import SeismicLithologyPredictor

predictor = SeismicLithologyPredictor(
    input_segy_list=input_segy_list,
    output_segy=output_segy,
    horizon_top=horizon_top,
    horizon_base=horizon_base,
    header_bytes=header_bytes,
    model_path="model_lith.pkl",   # from geosc.ml.Classifier.save(...)
    output_mode="labels",          # labels | prob_class | max_prob
    probability_class=1,           # required only for prob_class
    null_value=-999.25,
    t0=seis_time_first_sample,
    dt=seis_sample_interval,
    ns=seis_sample_pertrace,
)
predictor.run()
```

### Seismic Classification Workflow

```python
from geosc.seismic import SeismicClassifier

classifier = SeismicClassifier(
    input_segy_list=input_segy_list,
    output_segy=output_segy,
    horizon_top=horizon_top,
    horizon_base=horizon_base,
    header_bytes=header_bytes,
    model_path="model_classification.pkl",  # from geosc.ml.Classifier.save(...)
    output_mode="labels",                   # labels | prob_class | max_prob
    probability_class=1,                    # required only for prob_class
    null_value=-999.25,
    t0=seis_time_first_sample,
    dt=seis_sample_interval,
    ns=seis_sample_pertrace,
)
classifier.run()
```

### Seismic Regression Workflow

```python
from geosc.seismic import SeismicRegressor

regressor = SeismicRegressor(
    input_segy_list=input_segy_list,
    output_segy=output_segy,
    horizon_top=horizon_top,
    horizon_base=horizon_base,
    header_bytes=header_bytes,
    model_path="model_regression.pkl",      # from geosc.ml.Regressor.save(...)
    null_value=-999.25,
    t0=seis_time_first_sample,
    dt=seis_sample_interval,
    ns=seis_sample_pertrace,
)
regressor.run()
```

## Usage Examples

See `scripts/` directory for complete examples:

- `test_attribute_*.py` - Seismic attribute examples
- `test_WellLog_regression.py` - Regression workflow
- `test_WellLog_clustering.py` - Unsupervised clustering
- `test_WellLog_litho.py` - Lithology prediction
- `test_seismic_clustering_2d.py` / `test_seismic_clustering_3d.py` - Seismic clustering
- `test_seismic_lithology_2d.py` / `test_seismic_lithology_3d.py` - Seismic lithology prediction
- `test_seismic_regression_2d.py` / `test_seismic_regression_3d.py` - Seismic regression prediction
- `plot_seismic_clustering_2d.py` / `plot_seismic_clustering_3d.py` - Plot clustering results
- `plot_seismic_lithology_2d.py` / `plot_seismic_lithology_3d.py` - Plot lithology with custom class colors
- `plot_seismic_regression_2d.py` - Plot regression results
- `plot_naive_bayes_gaussian_contours_2d.py` - Generic 2D Gaussian NB contour plot

### Running Examples

```bash
cd /Drive/D/geoscience
pip install -e .
python scripts/test_attribute_2d.py
python scripts/test_WellLog_litho.py
```

## Adding New Features

### Adding a Seismic Attribute

1. Create `geosc/seismic/attributes/myattribute.py`
2. Inherit from `TraceAttribute` or `WindowAttribute`
3. Implement `process_trace()` or `process_window()`
4. Update `geosc/seismic/attributes/__init__.py`
5. Create test script in `scripts/`

### Adding Gridding/Interpolation

File: `geosc/map/gridding/utils.py`
- `idw_interpolate(x, y, values, grid_x, grid_y, power=2)`
- `auto_variogram(x, y, values, model="spherical")`

### Accessing SEG-Y Headers

File: `geosc/seismic/segy/trace_headers.py`
- Use `get_segy_trace_headers()` for batch reading
- Pass explicit byte positions for coordinate arrays

## Contributing

Before contributing:
1. Test locally using scripts in `scripts/` directory
2. Verify base class handling works (I/O, memory options)
3. Ensure numerical stability (NaN handling, edge cases)
4. Use `DataCleaner` for ML workflows
5. Add docstrings and examples for new features

## Requirements for Development

```bash
pip install -e ".[dev]"
```

See `setup.py` for full dependency list.

## License

[Add your license here]

## Contact & Support

For issues, feature requests, or questions, please open an issue on GitHub.

---

**Happy Processing!** 🎉
