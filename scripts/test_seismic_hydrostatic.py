from geosc.seismic.geomechanics import SeismicHydrostatic
from geosc.seismic.segy import append_process_text_header


# Input depth TVDSS volume (meter)
input_segy = "/Drive/D/Temp/depth_tvdss.sgy"

# Output hydrostatic in psi
output_segy_psi = "/Drive/D/Temp/hydrostatic_psi.sgy"

calculator_psi = SeismicHydrostatic(
    input_segy=input_segy,
    output_segy=output_segy_psi,
    output_unit="psi",  # "psi" or "sg"
    density=1000.0,     # fluid density [kg/m^3]
    gravity=9.78033,    # gravitational acceleration [m/s^2]
    null_value=-999.25, # set None if input has no null marker
)
calculator_psi.run()
append_process_text_header(
    output_segy_psi,
    process_name="SeismicHydrostatic",
    details=[
        "output_unit=psi",
        "density_kgm3=1000.0",
        "gravity_mps2=9.78033",
        "null_value=-999.25",
    ],
)


# Output hydrostatic equivalent gradient in SG
output_segy_sg = "/Drive/D/Temp/hydrostatic_sg.sgy"

calculator_sg = SeismicHydrostatic(
    input_segy=input_segy,
    output_segy=output_segy_sg,
    output_unit="sg",
    density=1000.0,     # [kg/m^3]
    gravity=9.78033,    # [m/s^2]
    null_value=-999.25, # or None
)
calculator_sg.run()
append_process_text_header(
    output_segy_sg,
    process_name="SeismicHydrostatic",
    details=[
        "output_unit=sg",
        "density_kgm3=1000.0",
        "gravity_mps2=9.78033",
        "null_value=-999.25",
    ],
)
