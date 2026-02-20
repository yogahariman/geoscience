from geosc.seismic import crop_segy_3d
from geosc.seismic.segy import append_process_text_header

file_in = "/Drive/D/Works/DataSample/Seismic3D/Sample01/SE-WALIO-PURE-IRIS.sgy"
file_out = "/Drive/D/out_crop_100_200_100_200_t0_500.sgy"

crop_segy_3d(
    input_segy=file_in,
    output_segy=file_out,
    inline_range=(100, 200),
    xline_range=(100, 200),
    time_range=(0.0, 500.0),
    header_bytes={
        "INLINE": (25, "int32"),
        "XLINE": (29, "int32"),
    },
)

append_process_text_header(
    file_out,
    process_name="Crop3D",
    details=[
        f"input={file_in}",
        "inline_range=(100, 200)",
        "xline_range=(100, 200)",
        "time_range_ms=(0.0, 500.0)",
    ],
)
