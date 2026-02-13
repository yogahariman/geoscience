# contoh well log clustering (unsupervised)
# 1. preprocessing: bersihkan null pada X
# 2. clustering

import numpy as np
import pandas as pd

from geosc.ml import Clusterer, DataCleaner

NULL = -999.25

FEATURE_COLS = ["AI", "SI"]
DEPTH_COL = "DepthMD"
LABEL_OFFSET = 1

# -------- clustering --------

df = pd.read_csv(
    "/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/WellLog/WellLogCNOOC_CNOOC-2D.csv",
    header=0,
    skiprows=[1],
)
X = df[FEATURE_COLS].values

cleaner = DataCleaner(null_value=NULL)
X = cleaner.clean_data_prediction(X)

clusterer = Clusterer(model_type="som")
labels = clusterer.fit_predict(
    X,
    parameters=dict(n_rows=3, n_cols=3, n_iter=1000, learning_rate=0.5, sigma=2.0),
    scale_x=True,
    null_value=NULL,
)

# clusterer = Clusterer(model_type="kmeans")
# labels = clusterer.fit_predict(
#     X,
#     parameters=dict(n_clusters=5, random_state=42),
#     scale_x=True,
#     null_value=NULL,
# )

clusterer.save("/Drive/D/Temp/model_cluster.pkl")

labels = labels.astype(float)
labels = np.where(np.isnan(labels), NULL, labels)
valid_mask = labels != NULL
labels[valid_mask] = labels[valid_mask] + LABEL_OFFSET

df_out = pd.DataFrame({
    "cluster_label": labels
})

if DEPTH_COL in df.columns:
    df_out.insert(0, DEPTH_COL, df[DEPTH_COL].values)


df_out.to_csv("/Drive/D/Temp/output_cluster.csv", index=False)
