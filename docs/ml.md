# ML Guide

Dokumen ini menjelaskan alur ML yang sudah diimplementasikan di project ini.

## Modul Utama

- `geosc.ml`
  - `DataCleaner`: cleaning data training/prediksi
  - `Classifier`: supervised classification (xgboost, random_forest, mlp, svm, naive_bayes)
  - `Regressor`: supervised regression
  - `Clusterer`: unsupervised clustering
- `geosc.seismic.ml`
  - `SeismicClusterer`: clustering seismic dalam interval horizon
  - `SeismicLithologyPredictor`: prediksi lithology seismic (predict-only dari model `Classifier`)
- `geosc.plotting`
  - `NaiveBayesGaussianContour2D`: plotting Gaussian contour NB 2D (generic tabular)

## Prinsip Workflow

1. Load data dilakukan di luar modul (CSV/TXT/DB/SEG-Y bebas).
2. Alur umum: `training -> save model -> prediction`.
3. Null handling konsisten dengan `null_value` (default `-999.25`).

## Workflow Classification (Well/Tabular)

```python
from geosc.ml import DataCleaner, Classifier

cleaner = DataCleaner(null_value=-999.25)
X_train, y_train = cleaner.clean_data_training(X, y)

model = Classifier(model_type="mlp")
model.train(X_train, y_train, parameters={"max_iter": 1000}, scale_x=True)
model.save("model_lith.pkl")

X_pred = cleaner.clean_data_prediction(X_pred)
y_pred, y_prob = model.predict(X_pred, null_value=-999.25)
```

## Workflow Seismic Lithology Prediction

`SeismicLithologyPredictor` memakai model hasil `Classifier.save(...)`.

Input utama:
- `input_segy_list`: 1..N volume atribut seismic
- `header_bytes`: mapping header (`X`, `Y`, plus `INLINE+XLINE` atau `CDP`)
- `horizon_top` / `horizon_base`: array `Nx3 [x, y, t]`
- `model_path`: file model `.pkl`

```python
from geosc.seismic import SeismicLithologyPredictor

predictor = SeismicLithologyPredictor(
    input_segy_list=input_segy_list,
    output_segy=output_segy,
    horizon_top=horizon_top,
    horizon_base=horizon_base,
    header_bytes=header_bytes,
    model_path="model_lith.pkl",
    output_mode="labels",   # labels | prob_class | max_prob
    probability_class=1,      # wajib jika output_mode="prob_class"
    null_value=-999.25,
)
predictor.run()
```

### `output_mode` pada Seismic Lithology

- `labels`
  - Output SEG-Y berisi label class hasil prediksi.
- `prob_class`
  - Output SEG-Y berisi probabilitas untuk satu class tertentu (`probability_class`).
- `max_prob`
  - Output SEG-Y berisi probabilitas maksimum antar semua class (confidence) untuk QC.

## Contoh Script

- `scripts/test_WellLog_classification.py`
- `scripts/test_WellLog_regression.py`
- `scripts/test_WellLog_clustering.py`
- `scripts/test_WellLog_litho.py`
- `scripts/test_seismic_clustering_2d.py`
- `scripts/test_seismic_clustering_3d.py`
- `scripts/test_seismic_lithology_prediction_2d.py`
- `scripts/test_seismic_lithology_prediction_3d.py`
- `scripts/plot_seismic_clustering_2d.py`
- `scripts/plot_seismic_clustering_3d.py`
- `scripts/plot_seismic_lithology_2d.py`
- `scripts/plot_seismic_lithology_3d.py`
- `scripts/plot_naive_bayes_gaussian_contours_2d.py`

## Plotting Lithology (Custom Class Colors)

Helper plotting lithology ada di:
- `geosc/seismic/plotting/lithology_utils.py`

Fungsi utama:
- `build_discrete_style(values, class_colors)`
- `mask_null_values(data, null_value)`

Contoh mapping warna:

```python
CLASS_COLORS = {
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "black",
}
```

Catatan:
- Jika ada class yang belum didefinisikan di `CLASS_COLORS`, warna fallback dari `tab20` akan dipakai.
