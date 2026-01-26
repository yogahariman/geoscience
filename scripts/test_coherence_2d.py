from geosc.seismic.attributes import Coherence2D
from geosc.seismic.plotting.plot_seismic_2d import SeismicPlot2D
from geosc.seismic.plotting import SeismicPlot2D

segyinput = "/Drive/D/Works/DataSample/Seismic2D/Sample01/81.21_Orig.sgy"
segyoutput = "/Drive/D/coherence_2d.sgy"

Coherence2D(
    segyinput, segyoutput,
    window=(20, 5),
    load_to_ram=True
).run()


plot2d = SeismicPlot2D(segyoutput)

# coherence
plot2d.plot(plot_type="coherence")
# plot2d.plot_coherence()