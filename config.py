# =============================================================================
# 
# ASTRA Configuration File
# 
# =============================================================================

# data set

dataset = 0

height = 1.7 # km

# select your file directory
data_directory  = 'data/datasets/new*'+str(dataset)+'/'
column_data = 'data/columndata/'
fluxes = 'data/truefluxes/'

# select the spectrograph order
spectrograph_order = 2

if spectrograph_order == 1:
    lower_wave = 5000 # lower wavelength cut off
    upper_wave = 10000
elif spectrograph_order == 2:
    lower_wave = 3250
    upper_wave = 6200

# select stars to remove
removed_stars  = []

residual = 0.00001
count_threshold = 5