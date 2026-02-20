# contoh seismic shmax prediction 2D (supervised)
# - model dari geosc.ml.ShmaxRegressor.save(...)
# - urutan fitur wajib sama dengan training model
# - contoh urutan: [hydrostatic, overburden, porepressure, shmin, AI, SI]

import os

import pandas as pd
import segyio

from geosc.seismic import SeismicShmaxPredictor
from geosc.seismic.segy import append_process_text_header, get_segy_trace_header

nmline = [
    "04-001",
    "04-006",
]

atb = [
    ("/Drive/D/Temp/HYDROSTATIC_2D/", "_HYDRO.sgy"),
    ("/Drive/D/Temp/OVERBURDEN_2D/", "_OB.sgy"),
    ("/Drive/D/Temp/POREPRESSURE_2D/", "_PP.sgy"),
    ("/Drive/D/Temp/SHMIN_2D/", "_SHMIN.sgy"),
    ("/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/Seismic/AI/", "#Zp.sgy"),
    ("/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/Seismic/SI/", "#SI.sgy"),
]

input_segy_list = []
for line in nmline:
    row = []
    for base, suffix in atb:
        row.append(f"{base}{line}{suffix}")
    input_segy_list.append(row)

output_segy = [f"/Drive/D/Temp/20260218_{line}_SHMAX_PRED.sgy" for line in nmline]
model_path = "/Drive/D/Temp/model_shmax.pkl"

horizon_top_csv = "/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/Horizon/BRF.csv"
horizon_base_csv = "/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/Horizon/Basement+100.csv"
horizon_top = pd.read_csv(horizon_top_csv, header=None).iloc[:, :3].to_numpy(dtype=float)
horizon_base = pd.read_csv(horizon_base_csv, header=None).iloc[:, :3].to_numpy(dtype=float)

header_bytes = []
for _ in nmline:
    row = []
    for _ in atb:
        row.append(
            {
                "X": (73, "int32"),
                "Y": (77, "int32"),
                "CDP": (21, "int32"),
            }
        )
    header_bytes.append(row)

missing = []
for line in input_segy_list:
    for segyfile in line:
        if not os.path.exists(segyfile):
            missing.append(segyfile)
print("Missing seismic:", missing)
print("Model exists:", os.path.exists(model_path), model_path)

seis_time_first_sample = []
for line in input_segy_list:
    row = []
    for segyfile in line:
        t0_val = -get_segy_trace_header(segyfile, 105, "int16")[0]
        row.append(float(t0_val))
    seis_time_first_sample.append(row)

seis_sample_interval = []
seis_sample_pertrace = []
for line in input_segy_list:
    dt_row = []
    ns_row = []
    for segyfile in line:
        with segyio.open(segyfile, "r", ignore_geometry=True) as f:
            dt_row.append(segyio.tools.dt(f) / 1000.0)
            ns_row.append(int(f.samples.size))
    seis_sample_interval.append(dt_row)
    seis_sample_pertrace.append(ns_row)

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
for out_path in output_segy:
    append_process_text_header(
        out_path,
        process_name="SeismicShmaxPredictor",
        details=[
            f"model={model_path}",
            "feature_order=hydrostatic,overburden,porepressure,shmin,...",
            f"n_attrs={len(atb)}",
            "null_value=-999.25",
        ],
    )
