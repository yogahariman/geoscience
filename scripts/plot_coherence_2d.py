from geosc.seismic.plotting.plot_seismic_2d import SeismicPlot2D

plot2d = SeismicPlot2D("/Drive/D/coherence_2d.sgy")

# coherence
plot2d.plot(plot_type="coherence")

# # semblance (yang kamu minta)
# plot2d.plot(plot_type="semblance")


# # amplitude
# plot2d.plot(plot_type="amplitude")

# # shortcut
# plot2d.plot_semblance()


