# contoh well log classification (target kategori)
# 1. preprocessing: bersihkan null pada X dan y
# 2. training
# 3. predicting

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from geosc.ml import Classifier, DataCleaner

NULL = -999.25

FEATURE_COLS = ["vp", "vs"]
TARGET_COL = "litho_id"

# -------- training --------

df_train = pd.read_csv("/Drive/D/Works/DataSample/WellLog_CSV/data_training.csv")
X = df_train[FEATURE_COLS].values
y = df_train[TARGET_COL].values

cleaner = DataCleaner(null_value=NULL)
X, y = cleaner.clean_data_training(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

predictor = Classifier(model_type="mlp")
predictor.train(
    X_train, y_train,
    parameters=dict(hidden_layer_sizes=(128, 64, 32), max_iter=50000),
    scale_x=True
)
predictor.save("/Drive/D/Temp/model_lith.pkl")

pred_test, _ = predictor.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))

# -------- predict --------

df = pd.read_csv("/Drive/D/Works/DataSample/WellLog_CSV/data_training.csv")
X = df[FEATURE_COLS].values

X = cleaner.clean_data_prediction(X)

predictor = Classifier.load("/Drive/D/Temp/model_lith.pkl")
lith_pred, lith_prob = predictor.predict(X, null_value=NULL)

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
