import shutil

from geosc.seismic.segy import append_process_text_header, read_text_header

file_in = "/Drive/E/Senoro_202112/Seismic/TOMORI/SENORO_inverted1REV_FULL_TOMORI_Zp.sgy"
# file_out = "/Drive/D/out_text_header_demo.sgy"

print("=== INPUT TEXT HEADER (block 0) ===")
lines_in = read_text_header(file_in)
for line in lines_in:
    print(line)

# shutil.copyfile(file_in, file_out)
# append_process_text_header(
#     file_out,
#     process_name="TextHeaderTest",
#     details=[
#         f"input={file_in}",
#         "action=append text header test",
#     ],
# )

# print("\n=== OUTPUT TEXT HEADER AFTER APPEND (block 0) ===")
# lines_out = read_text_header(file_out)
# for line in lines_out:
#     print(line)
