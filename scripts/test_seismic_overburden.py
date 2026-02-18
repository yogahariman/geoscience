from geosc.seismic.geomechanics import SeismicOverburden


# Input volumes
# - TVDSS in meter
# - Density in kg/m^3
tvdss_segy = "/Drive/D/Temp/depth_tvdss.sgy"
density_segy = "/Drive/D/Temp/density.sgy"

output_segy_psi = "/Drive/D/Temp/overburden_psi.sgy"

header_bytes_tvdss = {
    "INLINE": (5, "int32"),  # for 3D
    "XLINE": (21, "int32"),  # for 3D
    # "CDP": (21, "int32"),  # use this for 2D
}

header_bytes_density = {
    "INLINE": (5, "int32"),
    "XLINE": (21, "int32"),
    # "CDP": (21, "int32"),
}

calc_psi = SeismicOverburden(
    tvdss_segy=tvdss_segy,
    density_segy=density_segy,
    output_segy=output_segy_psi,
    tvdss_header_bytes=header_bytes_tvdss,
    density_header_bytes=header_bytes_density,
    output_unit="psi",     # "psi" or "sg"
    density_unit="g/cc",   # "g/cc" or "kg/m3"
    gravity=9.78033,       # [m/s^2]
    null_value=-999.25,    # set None if no null marker
)
calc_psi.run()


output_segy_sg = "/Drive/D/Temp/overburden_sg.sgy"

calc_sg = SeismicOverburden(
    tvdss_segy=tvdss_segy,
    density_segy=density_segy,
    output_segy=output_segy_sg,
    tvdss_header_bytes=header_bytes_tvdss,
    density_header_bytes=header_bytes_density,
    output_unit="sg",
    density_unit="g/cc",
    gravity=9.78033,
    null_value=-999.25,
)
calc_sg.run()
