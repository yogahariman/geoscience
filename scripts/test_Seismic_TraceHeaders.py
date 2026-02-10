from wsgiref import headers
import numpy as np
import segyio
from geosc.seismic.segy import get_segy_trace_header, get_segy_trace_headers

segyfile = "/Drive/D/Works/DataSample/Seismic2D/Sample01/81.21_AI.sgy"

with segyio.open(segyfile, "r", ignore_geometry=True) as f:
    # header per trace (dict-like)
    h100 = f.header[100]          # trace header pertama
    print(h100)                 # semua field header yang dikenali segyio

    # # kalau mau semua trace:
    # headers = [f.header[i] for i in range(f.tracecount)]
breakpoint()  # Debugger akan berhenti di sini
print()

# t0 = -get_segy_trace_header(segyfile, 105, "int16")  # ms -LagTimeA
# print("Trace 105 Time of first sample (ms):", t0)

# # baca samples axis (time/depth)
# with segyio.open(segyfile, "r", ignore_geometry=True) as f:
#     samples = np.asarray(f.samples, dtype=float)
#     print("samples[:10] =", samples[:10])

# # baca X,Y dari header (contoh byte Petrel)
# headers = get_segy_trace_headers(
#     segyfile,
#     {
#         "X": (73, "int32"),
#         "Y": (77, "int32"),
#     }
# )

# print("X[:5] =", headers["X"][:5])
# print("Y[:5] =", headers["Y"][:5])
