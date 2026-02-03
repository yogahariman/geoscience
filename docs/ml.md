geoscience/
├── geosc/
│   ├── seismic/
│   │   ├── segy/
│   │   ├── attributes/
│   │   ├── plotting/
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── lithology/
│   │   │   ├── vsh/
│   │   │   ├── porosity/
│   │   │   ├── sw/
│   │   │   └── porepressure/
│   │
│   ├── well/
│   │   ├── las/
│   │   ├── logs/
│   │   └── ml/
│   │       ├── lithology/
│   │       ├── vsh/
│   │       ├── porosity/
│   │       ├── sw/
│   │       └── porepressure/
│   │
│   ├── map/
│   │   ├── gridding/
│   │   ├── geo/
│   │   └── ml/
│   │       ├── lithology/
│   │       ├── vsh/
│   │       ├── porosity/
│   │       ├── sw/
│   │       └── porepressure/
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── datasets/
│   │   │   ├── base.py
│   │   │   ├── seismic.py
│   │   │   ├── well.py
│   │   │   └── map.py
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── normalization.py
│   │   │   ├── scaler.py
│   │   │   └── splitter.py
│   │   │
│   │   ├── models/
│   │   │   ├── base.py
│   │   │   ├── classification/
│   │   │   │   ├── bpnn.py
│   │   │   │   ├── naive_bayes.py
│   │   │   │   └── svm.py
│   │   │   │
│   │   │   ├── regression/
│   │   │   │   ├── bpnn.py
│   │   │   │   └── linear.py
│   │   │   │
│   │   │   └── clustering/
│   │   │       ├── som.py
│   │   │       └── kmeans.py
│   │   │
│   │   ├── pipelines/
│   │   │   ├── classification.py
│   │   │   ├── regression.py
│   │   │   └── clustering.py
│   │   │
│   │   └── outputs/
│   │       ├── matrix.py
│   │       ├── seismic.py
│   │       ├── well.py
│   │       └── map.py
│   │
│   └── utils/
│       ├── io.py
│       ├── math.py
│       └── validation.py

1. untuk load data misal dari csv atau txt itu dilakukan diluar modul saja biar lebih flexsibel user mau ambil data darimana
2. konsepnya training->save model->prediksi
3. untuk klasifikasi naive bayes bisa dibuat Decision Boundary Plot terus output prediksi probabilitas dan class
3. data_training untuk kasus well,seismic, dan map itu sama berupa
    X = [atb1, atb2, atb3]
    y = target
4. data_prediksi well adalah X = [atb1, atb2, atb3]
5. data_prediksi map adalah X = [atb1, atb2, atb3]
6. untuk kasus seismic di hold dulu karena komplex    
7. data_prediksi seismic inputan berupa seismic maka
    a. parameter posisi inline, xline,x,y
    b. horizon atas [x, y, t]
    c. horizon bawah [x, y, t]
    d. samakan posisi tiap trace berdasarkan inline xline
    e. proses dilakukan per-trace dengan batas horizon atas dan horizon bawah

    headers = get_segy_trace_headers(
        "data.sgy",
        {
            "INLINE": (25, "int32"),
            "XLINE":  (29, "int32"),
            "X": (73, "int32"),
            "Y": (77, "int32"),
        }
    )
    INLINE = headers["INLINE"]
    XLINE  = headers["XLINE"]
    X      = headers["X"]
    Y      = headers["Y"]


classification
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import svc

from sklearn.neural_network import MLPRegressor
    
