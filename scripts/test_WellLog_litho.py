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

from geosc.well.ml import LithologyPredictor

NULL = -999.25

df_train = pd.read_csv("/Drive/D/Works/DataSample/WellLog_CSV/data_training.csv")
df_train.replace(NULL, np.nan, inplace=True)
df_train.dropna(inplace=True)

X = df_train[['vp','vs']].values
y = df_train['litho_id'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

predictor = LithologyPredictor(model_type="mlp")
predictor.train(
    X_train, y_train,
    parameters=dict(hidden_layer_sizes=(128,64,32), max_iter=50000)
)
predictor.save("model_lith.pkl")

pred_test, _ = predictor.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))

# -------- predict --------

df = pd.read_csv("/Drive/D/Works/DataSample/WellLog_CSV/data_training.csv")
df.replace(NULL, np.nan, inplace=True)
df.ffill(inplace=True)

depth = df['depth'].values
X = df[['vp','vs']].values

predictor = LithologyPredictor.load("model_lith.pkl")
lith_pred, lith_prob = predictor.predict(X)

# print(lith_pred)
lith_pred = lith_pred.astype(float)
lith_pred = np.where(np.isnan(lith_pred), NULL, lith_pred)
df_out = pd.DataFrame({
'depth': depth,
'lithology_pred': lith_pred
})

if lith_prob is not None:
    for idx in range(lith_prob.shape[1]):
        df_out[f"lithology_prob_{idx}"] = lith_prob[:, idx]


df_out.to_csv("output_lithology.csv", index=False)
