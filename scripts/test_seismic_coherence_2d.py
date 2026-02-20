from geosc.seismic.attributes import Coherence2D
from geosc.seismic.plotting import SeismicPlot2D
from geosc.seismic.segy import append_process_text_header

segyinput = "/Drive/D/Works/DataSample/Seismic2D/Sample01/81.21_Orig.sgy"
segyoutput = "/Drive/D/coherence_2d.sgy"

Coherence2D(
    segyinput, segyoutput,
    window=(20, 5),
    load_to_ram=True
).run()
append_process_text_header(
    segyoutput,
    process_name="Coherence2D",
    details=[
        f"input={segyinput}",
        "window=(20,5)",
        "load_to_ram=True",
    ],
)


plot2d = SeismicPlot2D(segyoutput)

# coherence
plot2d.plot(plot_type="coherence")
# plot2d.plot_coherence()
