# contoh seismic clustering 3D (unsupervised)
# - multi volume
# - horizon top/bottom (CSV -> matrix [x,y,time])
# - output SEG-Y cluster labels

import pandas as pd
import numpy as np
import segyio
from geosc.seismic.segy import append_process_text_header, get_segy_trace_header
from geosc.seismic import SeismicClusterer

input_segy_list = [
    "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Attribute/AI_tmt_exp.segy",
    "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Attribute/SI_tmt_exp.segy",
]

output_segy = "/Drive/D/Temp/cluster_labels_3d.sgy"

horizon_top_csv = "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Horizon/Horizon_top.csv"
horizon_base_csv = "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Horizon/Horizon_bottom.csv"

# horizon format: no header, fixed columns [x, y, z/time]
horizon_top = pd.read_csv(horizon_top_csv, header=None).iloc[:, :3].to_numpy(dtype=float)
horizon_base = pd.read_csv(horizon_base_csv, header=None).iloc[:, :3].to_numpy(dtype=float)

# header bytes per attribute (3D)
header_bytes = [
    {
        "X": (73, "int32"),
        "Y": (77, "int32"),
        "INLINE": (5, "int32"),
        "XLINE": (21, "int32"),
    },
    {
        "X": (73, "int32"),
        "Y": (77, "int32"),
        "INLINE": (5, "int32"),
        "XLINE": (21, "int32"),
    },
]

# time first sample from -LagTimeA (byte 105)
seis_time_first_sample = []
for segyfile in input_segy_list:
    t0_val = -get_segy_trace_header(segyfile, 105, "int16")[0]
    seis_time_first_sample.append(float(t0_val))

# karena firstime terbalik maka saya kalikan -1 untuk buatnya jadi positif (ms)
seis_time_first_sample = np.array(seis_time_first_sample, dtype=float) * -1.0  # convert to positive ms


# sample interval (dt) and samples per trace (ns) per seismic
seis_sample_interval = []
seis_sample_pertrace = []
for segyfile in input_segy_list:
    with segyio.open(segyfile, "r", ignore_geometry=True) as f:
        seis_sample_interval.append(segyio.tools.dt(f) / 1000.0)  # ms
        seis_sample_pertrace.append(int(f.samples.size))

cluster_params = {
    "scale_x": True,
    "n_rows": 2,
    "n_cols": 2,
    "n_iter": 1000,
    "learning_rate": 0.1,
    "sigma": None,  # default: max(n_rows, n_cols) / 2
    "random_state": 42,
}

model_output = "/Drive/D/Temp/model_cluster_3d.pkl"

clusterer = SeismicClusterer(
    input_segy_list=input_segy_list,
    output_segy=output_segy,
    horizon_top=horizon_top,
    horizon_base=horizon_base,
    header_bytes=header_bytes,
    sample_percent=60.0,
    null_value=-999.25,
    cluster_params=cluster_params,
    seed=42,
    model_type="som",
    t0=seis_time_first_sample,
    dt=seis_sample_interval,
    ns=seis_sample_pertrace,
    label_offset=1,
    model_output=model_output,
)

clusterer.run()
append_process_text_header(
    output_segy,
    process_name="SeismicClusterer",
    details=[
        "model_type=som",
        "feature_order=as_input_segy_list",
        f"n_attrs={len(input_segy_list)}",
        "null_value=-999.25",
    ],
)
