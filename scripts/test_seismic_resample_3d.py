from geosc.seismic import SeismicEditor
from geosc.seismic.segy import append_process_text_header

file_in = "/Drive/D/Works/DataSample/Seismic3D/Sample01/SE-WALIO-PURE-IRIS.sgy"
file_out = "/Drive/D/out_resample_il2_xl2.sgy"

inline_step = 2
xline_step = 2

SeismicEditor.resample_3d(
    input_segy=file_in,
    output_segy=file_out,
    inline_step=inline_step,
    xline_step=xline_step,
    header_bytes={
        "INLINE": (25, "int32"),
        "XLINE": (29, "int32"),
    },
)

append_process_text_header(
    file_out,
    process_name="Resample3D",
    details=[
        f"input={file_in}",
        f"inline_step={inline_step}",
        f"xline_step={xline_step}",
    ],
)
