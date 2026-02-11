# buat lithology prediction dengan cara pemanggilan seperti pada test_WellLog_litho.py
# 1. preporecessing data misal scaller data X kalo pakai mlp
# 2. training
# 3. predicting
#
# class bisa diakses oleh user selain python juga, misal user c#, c++, java, go, dll
# model_type = [
#     "xgboost",
#     "random_forest",
#     "mlp",
#     "svm",
#     "naive_bayes"
# ]

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from geosc.ml import Classifier, DataCleaner

NULL = -999.25
TRAIN_CLASSES = [1, 2, 3, 4, 5]

df_train = pd.read_csv(
    "/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/WellLog/WellLogCNOOC_CNOOC-2D.csv",
    header=0,
    skiprows=[1]
)

# pakai data training hanya untuk class lithology tertentu
df_train = df_train[df_train["LithoId"].isin(TRAIN_CLASSES)].copy()
if df_train.empty:
    raise ValueError(f"Tidak ada data training untuk class {TRAIN_CLASSES}.")

X = df_train[["AI", "SI"]].values
y = df_train["LithoId"].values

cleaner = DataCleaner(null_value=NULL)
X, y = cleaner.clean_data_training(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# predictor = Classifier(model_type="mlp")
# predictor.train(
#     X_train, y_train,
#     parameters=dict(hidden_layer_sizes=(128, 64, 32), max_iter=50000),
#     scale_x=True
# )
predictor = Classifier(model_type="random_forest")
predictor.train(
    X, y,
    parameters=dict(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    scale_x=False,  # random forest biasanya tidak perlu scaling
)
# predictor = Classifier(model_type="svm")
# predictor.train(
#     X_train, y_train,
#     parameters=dict(
#         C=10.0,
#         kernel="rbf",
#         gamma="scale",
#         class_weight="balanced",
#     ),
#     scale_x=True,  # penting untuk SVM
# )

# predictor = Classifier(model_type="naive_bayes")
# predictor.train(
#     X_train, y_train,
#     parameters=dict(
#         var_smoothing=1e-9
#     ),
#     scale_x=False,  # biasanya tidak wajib
# )

predictor.save("/Drive/D/Temp/model_lith.pkl")

# pred_test, _ = predictor.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, pred_test))
# print(classification_report(y_test, pred_test))

pred_test, _ = predictor.predict(X)
print("Accuracy:", accuracy_score(y, pred_test))
print(classification_report(y, pred_test))

# -------- predict --------

df = pd.read_csv("/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/WellLog/WellLogCNOOC_CNOOC-2D.csv")
X = df[["AI", "SI"]].values
X = cleaner.clean_data_prediction(X)

predictor = Classifier.load("/Drive/D/Temp/model_lith.pkl")
lith_pred, lith_prob = predictor.predict(X, null_value=NULL)

# print(lith_pred)
lith_pred = lith_pred.astype(float)
lith_pred = np.where(np.isnan(lith_pred), NULL, lith_pred)
df_out = pd.DataFrame({
    "lithology_pred": lith_pred
})

if "depth" in df.columns:
    df_out.insert(0, "depth", df["depth"].values)

if lith_prob is not None:
    for idx in range(lith_prob.shape[1]):
        df_out[f"lithology_prob_{idx}"] = lith_prob[:, idx]


df_out.to_csv("/Drive/D/Temp/output_lithology.csv", index=False)
