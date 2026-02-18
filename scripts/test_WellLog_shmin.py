# contoh well log shmin regression (special transform)
# aturan penting untuk X:
# - kolom pertama X wajib hydrostatic
# - kolom kedua X wajib overburden
# - kolom ketiga X wajib porepressure

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from geosc.ml import DataCleaner, ShminRegressor

NULL = -999.25

DATA_PATH = "/Drive/D/Temp/welllog_shmin.csv"
MODEL_PATH = "/Drive/D/Temp/model_shmin.pkl"
OUTPUT_PATH = "/Drive/D/Temp/output_shmin.csv"

# tiga pertama wajib [hydrostatic, overburden, porepressure], sisanya opsional
FEATURE_COLS = ["hydrostatic", "overburden", "porepressure", "AI", "SI"]
TARGET_COL = "shmin"

# dalam X: hydrostatic index 0, overburden index 1, porepressure index 2
HYDROSTATIC_COL_IN_X = 0
OVERBURDEN_COL_IN_X = 1
POREPRESSURE_COL_IN_X = 2

HAS_HEADER = True
read_header = 0 if HAS_HEADER else None

df_train = pd.read_csv(DATA_PATH, header=read_header)
for col in FEATURE_COLS + [TARGET_COL]:
    if col not in df_train.columns:
        raise ValueError(f"Kolom {col!r} tidak ditemukan di CSV.")

X = df_train[FEATURE_COLS].values
y = df_train[TARGET_COL].values

cleaner = DataCleaner(null_value=NULL)
X, y = cleaner.clean_data_training(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

predictor = ShminRegressor(
    model_type="mlp",
    porepressure_col=POREPRESSURE_COL_IN_X,
    hydrostatic_col=HYDROSTATIC_COL_IN_X,
    overburden_col=OVERBURDEN_COL_IN_X,
)
predictor.train(
    X_train,
    y_train,
    parameters=dict(hidden_layer_sizes=(128, 64, 32), max_iter=50000),
    scale_x=True,
    scale_y=True,
)
predictor.save(MODEL_PATH)

pred_test = predictor.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_test)))
print("R2:", r2_score(y_test, pred_test))

# -------- predict full --------
df_pred = pd.read_csv(DATA_PATH, header=read_header)
X_pred = df_pred[FEATURE_COLS].values
X_pred = cleaner.clean_data_prediction(X_pred)

predictor = ShminRegressor.load(MODEL_PATH)
shmin_pred = predictor.predict(X_pred, null_value=NULL)
shmin_pred = np.asarray(shmin_pred, dtype=float)
shmin_pred = np.where(np.isnan(shmin_pred), NULL, shmin_pred)

df_out = pd.DataFrame({"shmin_pred": shmin_pred})
df_out.to_csv(OUTPUT_PATH, index=False)
