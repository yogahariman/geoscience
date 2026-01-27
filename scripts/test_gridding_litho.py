import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from geosc.map.gridding.interpolate_class import grid_class


# =========================================================
# INPUT DATA
# =========================================================

# data titik (well / sample)
df_data = pd.read_csv(
    "/Drive/D/Works/DataSample/Gridding/data_dummy_slice.csv"
)

x = df_data["x"].values
y = df_data["y"].values
lith = df_data["lithology"].values   # <-- class / lithology


# grid target (UNSTRUCTURED, 1D)
df_grid = pd.read_csv(
    "/Drive/D/Works/DataSample/Gridding/data_dummy_grid_map_xy.csv"
)

gx = df_grid["x"].values
gy = df_grid["y"].values


# =========================================================
# GRIDDING CLASS (KRIGING)
# =========================================================

dx = x.max() - x.min()
dy = y.max() - y.min()

domain_size = max(dx, dy)

range_kriging = 0.4 * domain_size


class_map_1d, probmaps = grid_class(
    x, y, lith,
    gx, gy,
    # method="idw",   # "nearest" / "idw" / "kriging"
    # power=2        # default = 2
    method="kriging",
    variogram_model="spherical",
    variogram_parameters={
        "range": range_kriging,
        "sill": 1.0,
        "nugget": 0.05
    },
    universal=False
)

# class_map_1d -> hasil class di gx, gy (1D)


# =========================================================
# WRITE RESULT TO CSV
# =========================================================

# df_out = pd.DataFrame({
#     "x": gx,
#     "y": gy,
#     "lithology": class_map_1d
# })

# df_out.to_csv("lithology_grid_result.csv", index=False)

# print("CSV saved: lithology_grid_result.csv")


# =========================================================
# PREPARE GRID FOR PLOTTING (DISPLAY ONLY)
# =========================================================

nx, ny = 500, 500   # resolusi display (AMAN RAM)

xu = np.linspace(gx.min(), gx.max(), nx)
yu = np.linspace(gy.min(), gy.max(), ny)

GX, GY = np.meshgrid(xu, yu)


# interpolasi ringan untuk visualisasi
class_map_2d = griddata(
    (gx, gy),
    class_map_1d,
    (GX, GY),
    method="nearest"   # class → nearest PALING BENAR
)


# =========================================================
# PLOT CLASS MAP
# =========================================================

plt.figure(figsize=(9, 7))

im = plt.pcolormesh(
    GX, GY, class_map_2d,
    shading="auto",
    cmap="tab20"
)

plt.scatter(x, y, c=lith, s=10, cmap="tab20", edgecolor="k")

cbar = plt.colorbar(im)
cbar.set_label("Lithology Class")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Lithology Map (Kriging Classification)")
plt.axis("equal")
plt.tight_layout()
plt.show()
