# contoh well log clustering (unsupervised)
# 1. preprocessing: bersihkan null pada X
# 2. clustering

import numpy as np
import pandas as pd

from geosc.ml import Clusterer, DataCleaner

NULL = -999.25

FEATURE_COLS = ["vp", "vs"]

# -------- clustering --------

df = pd.read_csv("/Drive/D/Works/DataSample/WellLog_CSV/data_training.csv")
X = df[FEATURE_COLS].values

cleaner = DataCleaner(null_value=NULL)
X = cleaner.clean_data_prediction(X)

clusterer = Clusterer(model_type="kmeans")
labels = clusterer.fit_predict(
    X,
    parameters=dict(n_clusters=5, random_state=42),
    scale_x=True,
    null_value=NULL,
)

clusterer.save("/Drive/D/Temp/model_cluster.pkl")

labels = labels.astype(float)
labels = np.where(np.isnan(labels), NULL, labels)

df_out = pd.DataFrame({
    "cluster_label": labels
})

if "depth" in df.columns:
    df_out.insert(0, "depth", df["depth"].values)


df_out.to_csv("/Drive/D/Temp/output_cluster.csv", index=False)
