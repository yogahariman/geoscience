# contoh well log SOM clustering (unsupervised)
# 1. preprocessing: bersihkan null pada X
# 2. som clustering

import numpy as np
import pandas as pd

from geosc.ml import Clusterer, DataCleaner

NULL = -999.25

FEATURE_COLS = ["vp", "vs"]

# -------- SOM clustering --------

df = pd.read_csv("/Drive/D/Works/DataSample/WellLog_CSV/data_training.csv")
X = df[FEATURE_COLS].values

cleaner = DataCleaner(null_value=NULL)
X = cleaner.clean_data_prediction(X)

clusterer = Clusterer(model_type="som")
labels = clusterer.fit_predict(
    X,
    parameters=dict(n_rows=5, n_cols=5, n_iter=1000, learning_rate=0.5, sigma=2.0),
    scale_x=True,
    null_value=NULL,
)

clusterer.save("/Drive/D/Temp/model_som.pkl")

labels = labels.astype(float)
labels = np.where(np.isnan(labels), NULL, labels)

# optional: mulai label dari 1
# labels = labels + 1

df_out = pd.DataFrame({
    "som_label": labels
})

if "depth" in df.columns:
    df_out.insert(0, "depth", df["depth"].values)


df_out.to_csv("/Drive/D/Temp/output_som.csv", index=False)
