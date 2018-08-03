import astra
import config

file_directory = config.data_directory
dataset = config.dataset
column_data = config.column_data
fluxes = config.fluxes
spectrograph_order = str(config.spectrograph_order)
lower_wave = int(config.lower_wave)
upper_wave = int(config.upper_wave)
removed_stars = set(config.removed_stars)
h = config.height

data, wavelength, airmass, time, weights, starlist, ext_stars, star,\
    observations, number_waves, increment = astra.load_data(file_directory, spectrograph_order, weighting=False)

astra.plot_bouguer(data, wavelength, airmass, starlist, star, ext_stars)