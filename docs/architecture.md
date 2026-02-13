# Architecture

Struktur implementasi saat ini (ringkas):

```text
geoscience/
├── geosc/
│   ├── __init__.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── classification.py
│   │   ├── regression.py
│   │   ├── clustering.py
│   │   └── lithology.py
│   ├── plotting/
│   │   ├── __init__.py
│   │   └── classification_2d.py
│   ├── seismic/
│   │   ├── __init__.py
│   │   ├── segy/
│   │   ├── attributes/
│   │   ├── plotting/
│   │   └── ml/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── classification.py
│   │       ├── regression.py
│   │       ├── clustering.py
│   │       └── lithology.py
│   ├── map/
│   │   └── gridding/
│   └── well/
│       └── __init__.py
├── scripts/
├── docs/
├── pyproject.toml
└── README.md
```

## Layering

- Core ML (`geosc.ml`)
  - Menangani training/predict model umum (tabular) + cleaning data.
- Generic plotting (`geosc.plotting`)
  - Plot reusable lintas domain (well/map/XRD/tabular).
  - Contoh: `NaiveBayesGaussianContour2D`.
- Domain Seismic ML (`geosc.seismic.ml`)
  - Menangani alignment trace antar-volume, pembatasan interval horizon, dan tulis output SEG-Y.
  - Memakai model dari `geosc.ml` (mis. `Classifier.load`).
- Shared Seismic ML Base (`geosc.seismic.ml.base`)
  - Fondasi bersama untuk validasi input, alignment trace, nearest horizon time, dan mapping sample axis (`t0`, `dt`, `ns`).
- Seismic Classification API (`geosc.seismic.ml.classification`)
  - API publik classification seismic generik berbasis `geosc.ml.Classifier`.
- Seismic Regression API (`geosc.seismic.ml.regression`)
  - API publik regression seismic generik berbasis `geosc.ml.Regressor`.
- Script layer (`scripts/`)
  - Contoh penggunaan end-to-end untuk 2D/3D dan well/seismic.
- Seismic plotting utils (`geosc.seismic.plotting`)
  - Plot 2D/3D umum: `SeismicPlot2D`, `SeismicPlot3D`
  - Util lithology discrete colors: `lithology_utils.py`

## Seismic ML Data Flow

1. Baca trace headers (`X`,`Y`,`INLINE/XLINE` atau `CDP`) dari semua volume.
2. Align trace berdasarkan key geometri yang beririsan.
3. Cocokkan `horizon_top/base` via nearest `X,Y`.
4. Bangun fitur multi-atribut per sample dalam interval horizon.
5. Jalankan model:
   - clustering: fit + predict
   - lithology: load classifier + predict
   - regression: load regressor + predict
6. Tulis output ke SEG-Y template (copy dari volume referensi).

## Public API (seismic)

- `from geosc.seismic import SeismicClusterer`
- `from geosc.seismic import SeismicLithologyPredictor`
- `from geosc.seismic import SeismicClassifier`
- `from geosc.seismic import SeismicRegressor`
