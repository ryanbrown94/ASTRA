# ASTRA

This software package is designed to correct the effects of atmopsheric extinction on stellar flux data.

##Functions

###Load Data

```python
load_data(file_directory, spectrograph_order):
#
#
#
returns (data, wavelength, airmass, time, weights, starlist,
ext_stars, star, observations, number_waves, increment)
```

* **data:** array of photon counts per second
* **wavelength:** array of wavelengths corresponding to *data*
* **airmass:** airmass value for each observation
* **time:** timestamp of each observation
* **weights:** array of appropriate weights (weighted by raw counts)
* **starlist:** returns the star identifier for each observation
* **ext_stars:** returns the star identifier of all considered stars
* **star:** returns the observation index for all star identifiers
* **observations:** number of observations in the dataset
* **number_waves:** number of wavelength data points
* **increment:** the wavelength separation (in Angstroms)

