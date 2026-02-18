# contoh well log regression (target kontinu)
# 1. preprocessing: bersihkan null pada X dan y
# 2. training
# 3. predicting

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from geosc.ml import Regressor, DataCleaner

NULL = -999.25

DATA_PATH = "/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/WellLog/WellLogCNOOC_CNOOC-2D.csv"
FEATURE_COLS = ["AI", "SI"]
TARGET_COL = "VSH"
DEPTH_COL = "DepthMD"

# -------- training --------

df_train = pd.read_csv(
    DATA_PATH,
    header=0,
    skiprows=[1],
)
X = df_train[FEATURE_COLS].values
y = df_train[TARGET_COL].values

cleaner = DataCleaner(null_value=NULL)
X, y = cleaner.clean_data_training(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

predictor = Regressor(model_type="mlp")
predictor.train(
    X_train, y_train,
    parameters=dict(hidden_layer_sizes=(128, 64, 32), max_iter=50000),
    scale_x=True,
    scale_y=True,
)
predictor.save("/Drive/D/Temp/model_vsh.pkl")

pred_test = predictor.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred_test))
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
print("RMSE:", rmse)
print("R2:", r2_score(y_test, pred_test))

# -------- predict --------

df = pd.read_csv(DATA_PATH, header=0, skiprows=[1])
X = df[FEATURE_COLS].values

X = cleaner.clean_data_prediction(X)

predictor = Regressor.load("/Drive/D/Temp/model_vsh.pkl")
vsh_pred = predictor.predict(X, null_value=NULL)

vsh_pred = vsh_pred.astype(float)
vsh_pred = np.where(np.isnan(vsh_pred), NULL, vsh_pred)

df_out = pd.DataFrame({
    "vsh_pred": vsh_pred
})

if DEPTH_COL in df.columns:
    df_out.insert(0, DEPTH_COL, df[DEPTH_COL].values)


df_out.to_csv("/Drive/D/Temp/output_vsh.csv", index=False)
