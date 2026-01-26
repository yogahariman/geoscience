from geosc.seismic.attributes import InstantaneousAmplitude
from geosc.seismic.plotting import SeismicPlot2D

segyinput = "/Drive/D/Works/DataSample/Seismic2D/Sample01/81.21_Orig.sgy"
segyoutput = "/Drive/D/out.sgy"

InstantaneousAmplitude(segyinput, segyoutput).run()

plot2d = SeismicPlot2D(segyoutput)

# amplitude
plot2d.plot(plot_type="amplitude")
# plot2d.plot_amplitude()