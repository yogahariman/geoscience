# contoh seismic clustering (unsupervised)
# - multi volume
# - horizon top/bottom (CSV -> matrix [x,y,time])
# - output SEG-Y cluster labels

import pandas as pd
import segyio

from geosc.seismic.segy import get_segy_trace_header

from geosc.seismic import SeismicClusterer

nmline = [
    "05tr-072",
    "05tr-076",
]

atb = [
    ("/Drive/D/Works/DataSample/Seismic2D/Sample02(Barito)/Seismic/", "_AI"),
    ("/Drive/D/Works/DataSample/Seismic2D/Sample02(Barito)/Seismic/", "_SI"),
    ("/Drive/D/Works/DataSample/Seismic2D/Sample02(Barito)/Seismic/", "_LR"),
    ("/Drive/D/Works/DataSample/Seismic2D/Sample02(Barito)/Seismic/", "_MR"),
]

# build input_segy_list: [nLines, nAttributes]
input_segy_list = []
for line in nmline:
    row = []
    for base, suffix in atb:
        row.append(f"{base}{line}{suffix}.sgy")
    input_segy_list.append(row)

output_segy = []
for line in nmline:
    output_segy.append(
        f"/Drive/D/Temp/2026021001_{line}_SOM_C3x3_AI,SI,LR,MR.sgy"
    )

import os
missing = []
for line in input_segy_list:
    for segyfile in line:
        if not os.path.exists(segyfile):
            missing.append(segyfile)

print("Missing:", missing)


horizon_top_csv = "/Drive/D/Works/DataSample/Seismic2D/Sample02(Barito)/Horizon/SB10.csv"
horizon_base_csv = "/Drive/D/Works/DataSample/Seismic2D/Sample02(Barito)/Horizon/SB2+300.csv"

# horizon format: no header, fixed columns [x, y, z/time]
horizon_top = pd.read_csv(horizon_top_csv, header=None).iloc[:, :3].to_numpy(dtype=float)
horizon_base = pd.read_csv(horizon_base_csv, header=None).iloc[:, :3].to_numpy(dtype=float)

# Petrel-style byte position (1-based) + format
# define header bytes per attribute (per input volume)
header_bytes = []
for _ in nmline:
    row = []
    for _ in atb:
        row.append(
            {
                "X": (73, "int32"),
                "Y": (77, "int32"),
                "CDP": (21, "int32"),
                # "INLINE": (21, "int32"),
                # "XLINE": (25, "int32"),
            }
        )
    header_bytes.append(row)

seis_time_first_sample = []
for line in input_segy_list:
    row = []
    for segyfile in line:
        # t0 = -LagTimeA (byte 105, int16)
        t0 = -get_segy_trace_header(segyfile, 105, "int16")[0]
        row.append(float(t0))
    seis_time_first_sample.append(row)

# sample interval (dt) and samples per trace (ns) - edit here if needed
_ref_segy = input_segy_list[0][0]
with segyio.open(_ref_segy, "r", ignore_geometry=True) as f:
    seis_sample_interval = segyio.tools.dt(f) / 1000.0  # ms
    seis_sample_pertrace = int(f.samples.size)

cluster_params = {
    "n_rows": 3,
    "n_cols": 3,
    "n_iter": 100,
    "learning_rate": 0.01,
    "sigma": None,
    "random_state": 42,
}

model_output = "/Drive/D/Temp/model_cluster.pkl"

clusterer = SeismicClusterer(
    input_segy_list=input_segy_list,
    output_segy=output_segy,
    horizon_top=horizon_top,
    horizon_base=horizon_base,
    header_bytes=header_bytes,
    sample_percent=60.0,  # sampling data training
    null_value=-999.25,
    cluster_params=cluster_params,
    seed=42,
    model_type="som",
    seis_time_first_sample=seis_time_first_sample,
    dt=seis_sample_interval,
    ns=seis_sample_pertrace,
    label_offset=1,
    model_output=model_output,
)

clusterer.run()
