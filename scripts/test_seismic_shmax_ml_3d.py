# contoh seismic shmax prediction 3D (supervised)
# - model dari geosc.ml.ShmaxRegressor.save(...)
# - urutan fitur wajib sama dengan training model
# - contoh urutan: [hydrostatic, overburden, porepressure, shmin, AI, SI]

import os

import numpy as np
import pandas as pd
import segyio

from geosc.seismic import SeismicShmaxPredictor
from geosc.seismic.segy import append_process_text_header, get_segy_trace_header

input_segy_list = [
    "/Drive/D/Temp/HYDROSTATIC_3D/hydrostatic_3d.sgy",
    "/Drive/D/Temp/OVERBURDEN_3D/overburden_3d.sgy",
    "/Drive/D/Temp/POREPRESSURE_3D/porepressure_3d.sgy",
    "/Drive/D/Temp/SHMIN_3D/shmin_3d.sgy",
    "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Attribute/AI_tmt_exp.segy",
    "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Attribute/SI_tmt_exp.segy",
]

output_segy = "/Drive/D/Temp/shmax_pred_3d.sgy"
model_path = "/Drive/D/Temp/model_shmax.pkl"

horizon_top_csv = "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Horizon/Horizon_top.csv"
horizon_base_csv = "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Horizon/Horizon_bottom.csv"
horizon_top = pd.read_csv(horizon_top_csv, header=None).iloc[:, :3].to_numpy(dtype=float)
horizon_base = pd.read_csv(horizon_base_csv, header=None).iloc[:, :3].to_numpy(dtype=float)

header_bytes = [
    {"X": (73, "int32"), "Y": (77, "int32"), "INLINE": (5, "int32"), "XLINE": (21, "int32")},
    {"X": (73, "int32"), "Y": (77, "int32"), "INLINE": (5, "int32"), "XLINE": (21, "int32")},
    {"X": (73, "int32"), "Y": (77, "int32"), "INLINE": (5, "int32"), "XLINE": (21, "int32")},
    {"X": (73, "int32"), "Y": (77, "int32"), "INLINE": (5, "int32"), "XLINE": (21, "int32")},
    {"X": (73, "int32"), "Y": (77, "int32"), "INLINE": (5, "int32"), "XLINE": (21, "int32")},
    {"X": (73, "int32"), "Y": (77, "int32"), "INLINE": (5, "int32"), "XLINE": (21, "int32")},
]

missing = [p for p in input_segy_list if not os.path.exists(p)]
print("Missing seismic:", missing)
print("Model exists:", os.path.exists(model_path), model_path)

seis_time_first_sample = []
for segyfile in input_segy_list:
    t0_val = -get_segy_trace_header(segyfile, 105, "int16")[0]
    seis_time_first_sample.append(float(t0_val))
seis_time_first_sample = np.array(seis_time_first_sample, dtype=float) * -1.0

seis_sample_interval = []
seis_sample_pertrace = []
for segyfile in input_segy_list:
    with segyio.open(segyfile, "r", ignore_geometry=True) as f:
        seis_sample_interval.append(segyio.tools.dt(f) / 1000.0)
        seis_sample_pertrace.append(int(f.samples.size))

predictor = SeismicShmaxPredictor(
    input_segy_list=input_segy_list,
    output_segy=output_segy,
    horizon_top=horizon_top,
    horizon_base=horizon_base,
    header_bytes=header_bytes,
    model_path=model_path,
    null_value=-999.25,
    t0=seis_time_first_sample,
    dt=seis_sample_interval,
    ns=seis_sample_pertrace,
)
predictor.run()
append_process_text_header(
    output_segy,
    process_name="SeismicShmaxPredictor",
    details=[
        f"model={model_path}",
        "feature_order=hydrostatic,overburden,porepressure,shmin,...",
        f"n_attrs={len(input_segy_list)}",
        "null_value=-999.25",
    ],
)
