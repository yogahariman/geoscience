import pandas as pd
import numpy as np
from geosc.map.gridding.interpolate_value import KrigingGridder
import matplotlib.pyplot as plt

# data to be gridded 
df_ai = pd.read_csv("/Drive/D/Works/DataSample/Gridding/data_dummy_slice.csv")
x = df_ai["x"].values
y = df_ai["y"].values
ai = df_ai["acoustic_impedance"].values

# grid target
df_grid = pd.read_csv("/Drive/D/Works/DataSample/Gridding/data_dummy_grid_map_xy.csv")
gx = df_grid["x"].values
gy = df_grid["y"].values

gridder = KrigingGridder(
    variogram_model="spherical",   # atau "exponential", "gaussian"
    universal=False                # Ordinary Kriging
)

# gridder = KrigingGridder(
#     variogram_model="spherical",
#     universal=True,
#     drift_terms=["regional_linear"]
# )

# gridder = KrigingGridder(
#     variogram_model="spherical",
#     variogram_parameters={
#         "sill": 1200,
#         "range": 800,
#         "nugget": 50
#     }
# )

ai_grid = gridder.predict(
    x, y, ai,
    gx, gy
)

# out = pd.DataFrame({
#     "x": gx.flatten(),
#     "y": gy.flatten(),
#     "ai": ai_grid.flatten()
# })

# out.to_csv("ai_kriging_grid.csv", index=False)

# Reshape grid data to 2D for plotting
gx = gx.reshape(-1, 1) if gx.ndim == 1 else gx
gy = gy.reshape(-1, 1) if gy.ndim == 1 else gy
ai_grid = ai_grid.reshape(-1, 1) if ai_grid.ndim == 1 else ai_grid

# Create meshgrid if needed
gx_mesh, gy_mesh = np.meshgrid(np.unique(gx), np.unique(gy))
ai_grid_mesh = ai_grid.reshape(gx_mesh.shape)
# Create map plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot gridded data as contour/heatmap
contour = ax.contourf(gx, gy, ai_grid, levels=20, cmap="viridis")

# Overlay original data points
ax.scatter(x, y, c=ai, s=30, cmap="viridis", edgecolors="black", linewidth=0.5, alpha=0.7, label="Sample Data")

# Colorbar
cbar = plt.colorbar(contour, ax=ax, label="AI Value")

# Labels and title
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_title("Acoustic Impedance (AI) Kriging Interpolation Map")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()