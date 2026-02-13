from geosc.seismic.plotting.plot_seismic_2d import SeismicPlot2D


segyfile = "/Drive/D/Temp/2026021101_04-001_VSH_PRED_AI,SI.sgy"
null_value = -999.25

plot2d = SeismicPlot2D(segyfile)
plot2d.plot(
    plot_type="vsh",
    null_value=null_value,
    title="Seismic VShale 2D",
)
