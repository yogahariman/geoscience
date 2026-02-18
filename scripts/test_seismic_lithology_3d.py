# contoh seismic lithology prediction 3D (supervised)
# - model diambil dari hasil training geosc.ml.classification (Classifier.save)
# - multi volume input
# - horizon top/bottom (CSV -> matrix [x,y,time])
# - output SEG-Y lithology labels

import os

import numpy as np
import pandas as pd
import segyio

from geosc.seismic import SeismicLithologyPredictor
from geosc.seismic.segy import get_segy_trace_header

input_segy_list = [
    "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Attribute/AI_tmt_exp.segy",
    "/Drive/D/Works/DataSample/Seismic3D/Sample03_TMT20121220/Attribute/SI_tmt_exp.segy",
]

output_segy = "/Drive/D/Temp/lithology_pred_3d.sgy"
model_path = "/Drive/D/Temp/model_lith.pkl"

# pilih output:
# - "labels"     -> tulis label lithology hasil prediksi
# - "prob_class" -> tulis probabilitas untuk 1 kelas tertentu
# - "max_prob"   -> tulis probabilitas maksimum antar semua kelas (confidence)
output_mode = "labels"
probability_class = 1  # dipakai hanya jika output_mode="prob_class"

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

missing = [p for p in input_segy_list if not os.path.exists(p)]
print("Missing seismic:", missing)
print("Model exists:", os.path.exists(model_path), model_path)

# time first sample from -LagTimeA (byte 105)
seis_time_first_sample = []
for segyfile in input_segy_list:
    t0_val = -get_segy_trace_header(segyfile, 105, "int16")[0]
    seis_time_first_sample.append(float(t0_val))    

# jika perlu, sesuaikan sign t0 dengan data Anda
seis_time_first_sample = np.array(seis_time_first_sample, dtype=float)

# karena firstime terbalik maka saya kalikan -1 untuk buatnya jadi positif (ms)
seis_time_first_sample = np.array(seis_time_first_sample, dtype=float) * -1.0  # convert to positive ms

# sample interval (dt) and samples per trace (ns) per seismic
seis_sample_interval = []
seis_sample_pertrace = []
for segyfile in input_segy_list:
    with segyio.open(segyfile, "r", ignore_geometry=True) as f:
        seis_sample_interval.append(segyio.tools.dt(f) / 1000.0)  # ms
        seis_sample_pertrace.append(int(f.samples.size))

predictor = SeismicLithologyPredictor(
    input_segy_list=input_segy_list,
    output_segy=output_segy,
    horizon_top=horizon_top,
    horizon_base=horizon_base,
    header_bytes=header_bytes,
    model_path=model_path,
    null_value=-999.25,
    output_mode=output_mode,
    probability_class=probability_class,
    t0=seis_time_first_sample,
    dt=seis_sample_interval,
    ns=seis_sample_pertrace,
)

predictor.run()
