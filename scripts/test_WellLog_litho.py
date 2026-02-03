import numpy as np
import pandas as pd

from geosc.well.ml.lithology import train_lithology, predict_lithology

NULL_VALUE = -999.25

df_datTrain = pd.read_csv("data_training.csv")
df_datTrain.replace(NULL_VALUE, np.nan, inplace=True)
df_datTrain.dropna(inplace=True)

X = df_datTrain[['ai', 'si', 'vp', 'vs']].values
y = df_datTrain['lithology'].values

class_pred, class_prob = train_lithology(
    X, y,
    model_type="xgboost",
    model_out="xgboost_model.pkl",
    parameters={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1
    }
)


df = pd.read_csv("data_welllog.csv")
depth = df['depth'].values


df.replace(NULL_VALUE, np.nan, inplace=True)
df.fillna(method='ffill', inplace=True)


X = df[['ai', 'si', 'vp', 'vs']].values

lith_pred, lith_prob = predict_lithology(
    X,
    model="xgboost_model.pkl"
)

df_out = pd.DataFrame({
'depth': depth,
'lithology_pred': lith_pred
})


df_out.to_csv("output_lithology.csv", index=False)
