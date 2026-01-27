import pandas as pd
import numpy as np
from geosc.map.gridding.interpolate_value import grid_value
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# data to be gridded 
df_ai = pd.read_csv("/Drive/D/Works/DataSample/Gridding/data_dummy_slice.csv")
x = df_ai["x"].values
y = df_ai["y"].values
ai = df_ai["acoustic_impedance"].values

# grid target
df_grid = pd.read_csv("/Drive/D/Works/DataSample/Gridding/data_dummy_grid_map_xy.csv")
gx = df_grid["x"].values
gy = df_grid["y"].values

dx = x.max() - x.min()
dy = y.max() - y.min()

domain_size = max(dx, dy)

range_kriging = 0.4 * domain_size

ai_grid = grid_value(
    x, y, ai,
    gx, gy,
    method="kriging",
    variogram_model="spherical",
    variogram_parameters={
        "range": range_kriging,
        "sill": np.var(ai),
        "nugget": 0.05 * np.var(ai),
    },
    universal=False
)

# out = pd.DataFrame({
#     "x": gx.flatten(),
#     "y": gy.flatten(),
#     "ai": ai_grid.flatten()
# })

# out.to_csv("ai_kriging_grid.csv", index=False)

xu = np.unique(gx)
yu = np.unique(gy)

GX, GY = np.meshgrid(xu, yu)

ai_map = griddata(
    (gx, gy),      # titik hasil kriging
    ai_grid,       # value hasil kriging
    (GX, GY),      # grid display
    method="linear"   # "nearest" / "cubic"
)

plt.figure(figsize=(8, 6))

pcm = plt.pcolormesh(
    GX, GY, ai_map,
    shading="auto",
    cmap="viridis"
)

# titik data asli
plt.scatter(x, y, c="k", s=10, label="Data")

plt.colorbar(pcm, label="Acoustic Impedance")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("AI Map (Kriging)")
plt.legend()
plt.tight_layout()
plt.show()
