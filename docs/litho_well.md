Well Log lithology
metode yang bisa dipilih bpnn(mlp), xgboost, svm, naive bayes

null_value = -999.25
data_training.csv [ai, si, vp, vs lithology]
data_predicting.csv [depth, ai, si, vp, vs]

outputnya : [depth, lithology prediksi]

masukkan kedalam struktur ini
geoscience/
├── geosc/
│   ├── seismic/
│   ├── well/

jadi misal saya mau pakai bisa panggil
fromg geosc.well.lithology.train misal

pada saat training modelnya disimpan untuk keperluan prediksi
data training dilakukan preprocessing sebelum di train misal pakai scaller

from geosc.well.lithology import train_lithology

train_lithology(
    csv_file="data_training.csv",
    model_type="xgboost",   # mlp | svm | naive_bayes | xgboost
    model_out="model_lith.pkl"
)

from geosc.well.lithology import predict_lithology

predict_lithology(
    csv_file="data_predicting.csv",
    model_file="model_lith.pkl",
    output_csv="hasil_prediksi.csv"
)