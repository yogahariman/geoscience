from geosc.seismic.plotting.plot_seismic_2d import SeismicPlot2D

plot2d = SeismicPlot2D("/Drive/D/Temp/2026021101_04-001_SOM_C3x3_AI,SI.sgy")

# clustering
plot2d.plot(plot_type="cluster", null_value=-999.25)
