import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import time as runtime
import config
from astropy.io import fits
from scipy import optimize, interpolate, stats

# load external configuration parameters

file_directory = config.data_directory
dataset = config.dataset
column_data = config.column_data
fluxes = config.fluxes
spectrograph_order = str(config.spectrograph_order)
lower_wave = int(config.lower_wave)
upper_wave = int(config.upper_wave)
removed_stars = set(config.removed_stars)
h = config.height


# define global functions


# convert photon counts to magnitude
def magnitude(photon_count):
    return -2.5*np.log10(photon_count) + 15.0


# convert magnitude back to flux
def inverse_magnitude(mag):
    return 10**((mag-15.0)/(-2.5))


# Bouguer equation
def bouguer(m_obs, k, x):
    return m_obs + k*x


# convert angstroms to microwaves
def micro_wave(wavel):
    return wavel/10000.0


# calculate Rayleigh scattering extinction
def rayleigh(wavel):
    refr = 0.23465 + (1.076E2/(146-(1./wavel)**2)) + (0.93161/(41-(1./wavel)**2))
    return 9.4977E-3 * ((1./wavel)**4) * (refr**2) * (np.exp(-h/7.996))


# calculate dust extinction
def dust(A0, wavel, a):
    return A0 * wavel**(-a) * np.exp(-h/1.5)


# calculate ozone extinction
def ozone(T, k):
    return 1.11 * T * k


# convert vacuum wavelengths to wavelengths in air
def airwave(vacwave, wavenumber):
    return vacwave / (1.0000834213 + 2406030.0 / (1.30E10 - wavenumber**2) + 15997.0 / (3.89E9 - wavenumber**2))


def color(x):  # assign each star a unique color

    n2c = {1: 'C0',  # dark blue
           2: 'C1',  # orange
           3: 'C2',  # green
           4: 'C3',  # red
           5: 'C4',  # purple
           6: 'C5',  # brown
           7: 'C6',  # pink
           8: 'C7',  # gray
           9: 'C8',  # yellow
           0: 'C9'}  # light blue

    return [n2c[int(str(number)[-1])] for number in x]


# begin program


def load_data(file_directory, spectrograph_order):
    files = glob.glob((file_directory + '*-' + spectrograph_order + '.fits'))

    filelist = []  # generate empty arrays to fill in for loop
    starlist = []
    data = []
    wavelength = []
    airmass = []
    time = []
    weights = []

    upper_bound = None
    lower_bound = None

    for file in files:
        print(str(file) + '\n', end='', flush=True)  # prints each file name as it loads
        hdulist = fits.open(file)  # opens *.fits file
        filelist.append(file)  # fills array with file names
        starlist.append(hdulist[0].header['OBJECT'])  # fills array with star names
        airmass.append(hdulist[0].header['AIRMASS'])  # fills array with airmass
        time.append(hdulist[0].header['SIDEREAL'])  # fills array with sidereal time
        xrefval = hdulist[0].header['CRVAL1']  # minimum wavelength
        xrefpix = hdulist[0].header['CRPIX1']  # reference pixel coordinates
        xinc = hdulist[0].header['CDELT1']  # spectrograph resolution
        raw_data = fits.getdata(file)  # loads raw data for manipulation
        data_tmp = raw_data / hdulist[0].header['EXPTIME']  # corrects for varying exposure times
        wave = xrefval + (np.arange(len(data_tmp)) + 1 - xrefpix) * xinc  # creates array of wavelength
        lower_argument = np.argwhere(wave < lower_wave)  # calculate arguments where wavelength is less than threshold
        if not lower_argument.any():
            lower_bound = np.argmin(wave)
        else:
            lower_bound = np.max(lower_argument)  # select max argument
        upper_argument = np.argwhere(wave > upper_wave)  # calculate arguments where wavelength is more than threshold
        if not upper_argument.any():
            upper_bound = np.argmax(wave)
        else:
            upper_bound = np.min(upper_argument)  # select min argument
        data_tmp = data_tmp[lower_bound:upper_bound]  # truncate wavelength array
        data.append(data_tmp)
        wavelength.append(xrefval + (np.arange(len(data_tmp)) + 1 - xrefpix) * xinc + (lower_bound * xinc))
        weights.append(raw_data[lower_bound:upper_bound])
        hdulist.close()

    increment = fits.open(files[0])[0].header['CDELT1']
    airmass = np.array(airmass)
    wavelength = np.array(wavelength)  # convert to numpy arrays
    data = np.array(data)
    weights = np.array(weights)

    stars = list(set(starlist))  # creates array of unique stars
    stars = np.char.replace(stars, 'Star', '')  # removes text string to convert to float
    stars = set(np.asfarray(stars))  # converts to set
    ext_stars = stars - set(removed_stars)  # set algebra to remove stars
    ext_stars = np.asfarray(list(ext_stars))  # convert to array
    stars = np.asfarray(list(stars))  # converts to array

    starlist = np.char.replace(starlist, 'Star', '')  # array showing which star corresponds to observation
    starlist = np.asfarray(starlist)

    if len(removed_stars) == 0:
        print('No stars removed\n')
    else:
        print('Removed Stars:', removed_stars, '\n')

    observations, number_waves = np.shape(wavelength)  # extracts observation and wavelength data

    star = {}

    for i in stars:
        star[i] = np.argwhere(starlist == i)
        star[i] = np.array(star[i])
        star[i] = star[i][:, 0]

    return data, wavelength, airmass, time, weights, starlist, ext_stars, star, observations, number_waves, increment
