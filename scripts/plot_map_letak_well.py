import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from geosc.seismic.segy import get_segy_trace_headers


# ==========================================================
# USER CONFIG
# ==========================================================
segyfile = "/Drive/E/Senoro_202112/Seismic/SENORO_LayersCode_KintomB.sgy"

# Sesuaikan byte sesuai data Anda (Petrel-style, 1-based)
header_map = {
    "X": (121, "int32"),
    "Y": (125, "int32"),
    "INLINE": (173, "int32"),
    "XLINE": (177, "int32"),
}

# Opsi input manual:
# wells_xy = [("W-01", 431250.0, 9278450.0), ("W-02", 432100.0, 9279000.0)]
# Jika kosong, script akan minta input interaktif
# wells_xy: list[tuple[str, float, float]] = []
wells_xy = [
            ("SNR-01", 441460.04, 9846551.55),
            ("SNR-02", 438860.3, 9846049.8),
            ("SNR-03", 433970.8, 9841401.3),
            ("SNR-04", 438114.51, 9843180.24),
            ("SNR-05", 440078, 9848817.3),
            ("SNR-06", 431658.05, 9841081.18),
            ("SNR-09", 438834.73, 9846044.75),
            ("SNR-11", 438779.91, 9846032.97),
            ("SNR-14", 433961.48, 9841354.17),
            ("SNR-15", 431819.66, 9840895.13)
]

# Plot every N-th seismic trace point to keep plotting responsive on large files
seismic_plot_step = 1
plot_inline_xline = True


def _input_wells_from_user() -> list[tuple[str, float, float]]:
    wells: list[tuple[str, float, float]] = []
    nwell = int(input("Jumlah well: "))
    for i in range(nwell):
        name = input(f"Nama well #{i+1}: ").strip() or f"WELL_{i+1}"
        x = float(input(f"X well #{i+1}: "))
        y = float(input(f"Y well #{i+1}: "))
        wells.append((name, x, y))
    return wells


headers = get_segy_trace_headers(segyfile, header_map)

x = np.asarray(headers["X"], dtype=float)
y = np.asarray(headers["Y"], dtype=float)
inline = np.asarray(headers["INLINE"], dtype=int)
xline = np.asarray(headers["XLINE"], dtype=int)

step = max(1, int(seismic_plot_step))
plot_idx = np.arange(0, x.size, step, dtype=int)
xp = x[plot_idx]
yp = y[plot_idx]
inlinep = inline[plot_idx]
xlinep = xline[plot_idx]

if not wells_xy:
    wells_xy = _input_wells_from_user()

well_names = [w[0] for w in wells_xy]
well_xy = np.asarray([[w[1], w[2]] for w in wells_xy], dtype=float)

# map each well (X,Y) -> nearest seismic trace -> (INLINE, XLINE)
xy_tree = cKDTree(np.column_stack([x, y]))
_, idx_near = xy_tree.query(well_xy, k=1)
idx_near = np.asarray(idx_near, dtype=int)

well_inline = inline[idx_near]
well_xline = xline[idx_near]

print("=== Well Mapping Result ===")
for i, nm in enumerate(well_names):
    print(
        f"{nm}: "
        f"X={well_xy[i,0]:.3f}, Y={well_xy[i,1]:.3f} -> "
        f"INLINE={int(well_inline[i])}, XLINE={int(well_xline[i])}"
    )


# ==========================================================
# PLOT
# ==========================================================
# Main request: gabung XY well + XY seismic pada satu plot
plt.figure(figsize=(8, 7))
plt.scatter(xp, yp, s=3, c="lightgray", alpha=0.7, label="Seismic traces")
plt.scatter(well_xy[:, 0], well_xy[:, 1], s=70, c="red", marker="^", label="Wells")
for i, nm in enumerate(well_names):
    plt.text(well_xy[i, 0], well_xy[i, 1], f" {nm}", fontsize=9)
plt.title("Well + Seismic Map in X-Y")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Optional: plot domain INLINE-XLINE
if plot_inline_xline:
    plt.figure(figsize=(8, 7))
    plt.scatter(xlinep, inlinep, s=3, c="lightgray", alpha=0.7, label="Seismic traces")
    plt.scatter(well_xline, well_inline, s=70, c="blue", marker="^", label="Wells (mapped)")
    for i, nm in enumerate(well_names):
        plt.text(well_xline[i], well_inline[i], f" {nm}", fontsize=9)
    plt.title("Well + Seismic Map in INLINE-XLINE")
    plt.xlabel("XLINE")
    plt.ylabel("INLINE")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
