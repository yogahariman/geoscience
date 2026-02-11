import numpy as np
import segyio
import matplotlib.pyplot as plt
from geosc.seismic.plotting.lithology_utils import (
    build_discrete_style,
    mask_null_values,
)


segyfile = "/Drive/D/Temp/2026021101_04-001_LITH_PRED_AI,SI.sgy"
null_value = -999.25

# warna class bisa Anda atur bebas
# contoh: merah, hijau, kuning, biru, hitam
CLASS_COLORS = {
    1: "green",
    2: "yellow",
    3: "black",
    4: "blue",
    5: "red",
}

with segyio.open(segyfile, "r", ignore_geometry=True) as f:
    data = np.stack([f.trace[i] for i in range(f.tracecount)]).astype(float)
    ns = f.samples.size
    dt_ms = segyio.tools.dt(f) / 1000.0

valid = data[data != null_value]
cmap, norm, classes = build_discrete_style(valid, CLASS_COLORS)
plot_data = mask_null_values(data, null_value)

plt.figure(figsize=(14, 6))
img = plt.imshow(
    plot_data.T,
    aspect="auto",
    cmap=cmap,
    norm=norm,
    extent=[0, data.shape[0], ns * dt_ms, 0],
)

cbar = plt.colorbar(img, ticks=classes)
cbar.set_label("Lithology Class")

plt.xlabel("Trace")
plt.ylabel("Time (ms)")
plt.title("Seismic Lithology 2D")
plt.tight_layout()
plt.show()
