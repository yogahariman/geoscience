# contoh seismic regression prediction 2D (supervised)
# - model diambil dari hasil training geosc.ml.regression (Regressor.save)
# - multi volume input
# - horizon top/bottom (CSV -> matrix [x,y,time])
# - output SEG-Y nilai prediksi kontinu (misal vsh)

import os

import pandas as pd
import segyio

from geosc.seismic import SeismicRegressor
from geosc.seismic.segy import get_segy_trace_header

nmline = [
"04-001",
"04-006",
# "04-007",
# "88-012",
# "88-013",
# "88-038",
# "88-041",
# "88-104",
# "88-183",
# "88-193",
# "88-268",
# "88-286",
# "89-049",
# "89-065",
# "89-067",
# "89-069",
# "89-073b",
# "89-075a",
# "89-108",
# "89-110",
# "89-114",
# "89-116",
# "89-118",
# "89-120",
# "89-124",
# "90-101",
# "90-174",
# "91-163",
# "91-165",
# "91-203",
# "91-225",
# "91-245",
# "91-256",
# "91-259",
# "91-271",
# "91-288",
# "91-310",
# "91-329a",
# "91-348",
# "91-476",
# "91-480",
# "91-482",
# "91-486",
# "91-492",
# "93-020a",
# "93-024",
# "93-035",
# "93-037",
# "93-071a",
# "93-073",
# "93-079",
# "93-101",
# "93-109",
# "93-135",
# "93-155b",
# "93-156",
# "93-175",
# "93-179",
# "93-189",
# "93-199",
# "93-266",
# "93-308",
# "93-336a",
# "93-340",
# "93-368a",
# "93-408",
# "93-414",
# "93-420",
# "93-432",
# "93-436",
# "93-462",
# "93-474",
# "93-482",
# "94-011",
# "94-034",
# "94-062",
# "94-087",
# "94-130",
# "96-138",
# "96-139",
# "96-140",
# "96-141",
# "96-142",
# "96-144",
# "96-145",
# "96-153",
# "96-155",
# "96-156",
# "96-197a",
# "96-203",
]

atb = [
    ("/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/Seismic/AI/", "#Zp"),
    ("/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/Seismic/SI/", "#SI"),
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
        f"/Drive/D/Temp/2026021101_{line}_VSH_PRED_AI,SI.sgy"
    )

missing = []
for line in input_segy_list:
    for segyfile in line:
        if not os.path.exists(segyfile):
            missing.append(segyfile)
print("Missing seismic:", missing)

model_path = "/Drive/D/Temp/model_vsh.pkl"
print("Model exists:", os.path.exists(model_path), model_path)

horizon_top_csv = "/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/Horizon/BRF.csv"
horizon_base_csv = "/Drive/D/Works/DataSample/Seismic2D/Sample04(CNOOC)/Horizon/Basement+100.csv"

# horizon format: no header, fixed columns [x, y, z/time]
horizon_top = pd.read_csv(horizon_top_csv, header=None).iloc[:, :3].to_numpy(dtype=float)
horizon_base = pd.read_csv(horizon_base_csv, header=None).iloc[:, :3].to_numpy(dtype=float)

# Petrel-style byte position (1-based) + format
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

# time first sample from -LagTimeA (byte 105)
seis_time_first_sample = []
for line in input_segy_list:
    row = []
    for segyfile in line:
        t0_val = -get_segy_trace_header(segyfile, 105, "int16")[0]
        row.append(float(t0_val))
    seis_time_first_sample.append(row)

# sample interval (dt) and samples per trace (ns) per seismic
seis_sample_interval = []
seis_sample_pertrace = []
for line in input_segy_list:
    dt_row = []
    ns_row = []
    for segyfile in line:
        with segyio.open(segyfile, "r", ignore_geometry=True) as f:
            dt_row.append(segyio.tools.dt(f) / 1000.0)  # ms
            ns_row.append(int(f.samples.size))
    seis_sample_interval.append(dt_row)
    seis_sample_pertrace.append(ns_row)

predictor = SeismicRegressor(
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
