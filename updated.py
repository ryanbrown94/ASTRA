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


# define functions


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
           10: 'C9'}  # light blue

    return [n2c[number] for number in x]


# load data and prepare arrays

files = sorted(glob.glob(file_directory + '*-' + spectrograph_order + '.fits'))  # loads all relevant data files

filelist = []  # generate empty arrays to fill in for loop
starlist = []
data = []
wavelength = []
airmass = []
time = []
weights = []

print('\nLoading data...\n')

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

print('\nDone!')

stars = list(set(starlist))  # creates array of unique stars
stars = np.char.replace(stars, 'Star', '')  # removes text string to convert to float
stars = set(np.asfarray(stars))  # converts to set
ext_stars = stars - set(removed_stars)  # set algebra to remove stars
ext_stars = np.asfarray(list(ext_stars))  # convert to array
stars = np.asfarray(list(stars))  # converts to array

print('\nStars:', np.round(stars, 0), '\n')

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

print(observations, 'observations, each over', number_waves, 'wavelengths \n')
print('Wavelength Range:', np.nanmin(wavelength), '-', np.nanmax(wavelength), 'Angstroms\n')
print('Wavelength Step:', increment, 'Angstroms\n')

for i in range(1, 275, 1):
    plt.plot(airmass[i], magnitude(data[i, 0]), 'o', c=color(starlist)[i])
for j in range(1, 11, 1):
    XX, YY, r, p, e = stats.linregress(airmass[star[j]], magnitude(data[star[j], 0]))
    plt.plot(airmass[star[j]], XX*airmass[star[j]] + YY, c=color(star)[j-1])
plt.gca().invert_yaxis()
plt.title('Bouguer Diagram (@ '+str(np.min(wavelength[0][lower_bound:upper_bound]))+'A)')
plt.xlabel('Airmass')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

k0b = {}
m0b = {}
for j in range(number_waves):
    for i in range(1, 11, 1):
        k0b[i, j], m0b[i, j], _, _, _ = stats.linregress(airmass[star[i]], magnitude(data[star[i], j]))

star_k = {}
star_mag = {}
for j in range(1, 11, 1):
    for i in range(0, 786, 1):
        star_k[j, i] = k0b[j, i]
        star_mag[j, i] = m0b[j, i]

star_k = np.fromiter(star_k.values(), dtype=float)
star_k = np.reshape(star_k, (10, 786))
star_mag = np.fromiter(star_mag.values(), dtype=float)
star_mag = np.reshape(star_mag, (10, 786))
star_k_avg = np.average(star_k, axis=0)

# set up fitting matrices

print('\nGenerating matrices...\n')

index = np.zeros((len(ext_stars), observations))  # collation index array
d_mat = np.zeros((observations, number_waves))  # data
X_mat = np.zeros(observations)  # airmass
t_mat = np.zeros(observations)  # time

for position1, item1 in enumerate(ext_stars):
        for position, item in enumerate(starlist):
            if item == item1:
                index[position1, position] = 1  # sets matrix element to 1 for corresponding stars
                d_mat[position] = magnitude(data[position])  # creates matrix of data
                X_mat[position] = airmass[position]  # creates matrix of airmass
                t_mat[position] = time[position]-time[0]  # creates matrix of time

# perform matrix fitting

A = np.vstack([X_mat, X_mat*t_mat, index]).T  # design matrix
b = d_mat
alpha = np.dot(A.T, A)
beta = np.dot(A.T, b)
cov = np.linalg.inv(alpha)
matrix_parameters, matrix_residuals, matrix_rank, matrix_s = np.linalg.lstsq(A, b, rcond=None)
matrix_sigma = np.sqrt(np.sum((d_mat-np.dot(A, matrix_parameters))**2)/(len(t_mat)-3))


# least square fit using matrix method as initial guess
def leastsq_function(parameters, *args):  # define lst sq fitting function

    k = parameters[0]  # reads corresponding column from result matrix
    dk = parameters[1]
    m = parameters[2:]
    X = args[0]
    t = args[1]
    y = args[2]
    a = args[3]
    yfit = k*X + dk*X*t + np.dot(m, a)
    return y - yfit


print('Performing least squares fitting...\n')

# masking

# mask1 = np.logical_or(wavelength < 4090, wavelength > 4140)
# mask2 = np.logical_or(wavelength < 4340, wavelength > 4390)
# mask3 = np.logical_or(wavelength < 4860, wavelength > 4910)
#
# mask = np.logical_and.reduce((mask1, mask2, mask3))
# mask = np.invert(mask)
#
# d_mat[mask] = False

m0 = np.zeros((len(ext_stars), number_waves))  # creates empty array of correct size
k0 = np.zeros((observations, number_waves))
dk0 = np.zeros((observations, number_waves))
resid0 = np.zeros((len(starlist), number_waves))  # empty array of residuals

start = runtime.time()

for i in range(number_waves):
    result = optimize.leastsq(leastsq_function, matrix_parameters[:, i],
                              args=(X_mat, t_mat, d_mat[:, i], index))
    # optimize lst sq function using params as initial guess (from matrix)
    k0[0, i] = result[0][0]
    dk0[0, i] = result[0][1]
    m0[:, i] = result[0][2:]
    resid0[:, i] = d_mat[:, i] - (k0[:, i]*X_mat + dk0[:, i]*X_mat*t_mat + np.dot(m0[:, i], index))
    # calculate residuals y - yfit

m0t = np.zeros((len(ext_stars), number_waves))
k0t = np.zeros((observations, number_waves))
dk0t = np.zeros((observations, number_waves))
resid0t = np.zeros((len(starlist), number_waves))

end = runtime.time()
print('----- ' + str(np.round(end - start, 4)) + ' seconds -----\n')

# k0[0][k0[0] < 1e-100] = np.nan  # allows interpolation to remove balmer lines
# k0[0] = pd.DataFrame(k0[0]).interpolate().values.ravel().tolist()
#
# dk0[0][dk0[0] < 1e-100] = np.nan  # allows interpolation to remove balmer lines
# dk0[0] = pd.DataFrame(dk0[0]).interpolate().values.ravel().tolist()

vacwave_oz, k_oz = np.loadtxt(column_data + 'ozone.dat', unpack=True)  # load in ozone file
vacwave_oz = vacwave_oz * 10  # convert nm to angstroms
k_oz = k_oz * 2.687E19  # convert cm^2/mol to cm^-1, Loschmidt constant in mol/cm^3
oz_wavenb = 1E8 / vacwave_oz  # calculate wavenumber in cm^-1
airwave_oz = airwave(vacwave_oz, oz_wavenb)  # convert wavelength
interwave = interpolate.interp1d(np.round(airwave_oz, 1), k_oz, fill_value='extrapolate')  # interpolate wavelength
# to correct scale
inter = interwave(wavelength[0])

microwave = micro_wave(wavelength)  # convert wavelength to microwave

ray = rayleigh(microwave[0])
oz = (1.11 * 0.33 * inter)


def fitfunc(wavel, ray_multi, oz_multi, A0, alp):
    return (ray_multi * ray) \
           + (oz_multi * oz) \
           + (A0 * wavel ** (-alp) * np.exp(-h / 1.5))


popt, pcov = optimize.curve_fit(fitfunc, microwave[0], k0[0],
                                bounds=((0, 0, 0.01, 0.5), (1, 1, 0.05, 2.5)))

perr = np.diagonal(pcov)

popt = np.round(popt, 4)
perr = np.round(perr, 5)

print('Coefficients are: \n')
print('Rayleigh multiplier = ', popt[0], '+/-', perr[0], '('+str(np.round(perr[0]/popt[0] * 100, 3))+'%)')
print('Ozone multiplier = ', popt[1], '+/-', perr[1], '('+str(np.round(perr[1]/popt[1] * 100, 3))+'%)')
print('A0 = ', popt[2], '+/-', perr[2], '('+str(np.round(perr[2]/popt[2] * 100, 3))+'%)')
print('Alpha = ', popt[3], '+/-', perr[3], '('+str(np.round(perr[3]/popt[3] * 100, 3))+'%)')

modfit = fitfunc(microwave[0], *popt)  # fitted model using calculated parameters

barrywave, barrymodel = np.loadtxt('data/extdata/data0001-' + spectrograph_order + '.ext', unpack=True, skiprows=27)
barrywave = barrywave[lower_bound:upper_bound]
barrymodel = barrymodel[lower_bound:upper_bound]
barrymodel = -2.5*np.log10(barrymodel)/2.3031

plt.plot(wavelength[0], modfit, 'C0', label='ModelwithFittedParameters')
plt.plot(wavelength[0], k0[0], 'C1', label='LstSqFit')
plt.plot(barrywave, barrymodel, 'C2', label='BarryModel')
plt.title('Extinction Curve')
plt.xlabel('Wavelength (A)')
plt.ylabel('Mag per Airmass')
plt.legend(loc=0)
plt.show()

# new_removed_stars = set()
# print('\nStar, Max Residual')
# for i in star:
#     print(i, ',', np.round(np.max(np.abs(resid0[star[i]])),3))
#     if np.max(np.abs(resid0[star[i]])) > 0.1:
#         new_removed_stars = {i} ^ new_removed_stars
#
# print('\nNew removed stars are: ', new_removed_stars)

# TODO Finish removal criteria

averaged_data = np.average(d_mat, axis=1)
for i in range(1, 11, 1):
    plt.plot(t_mat[star[i]], averaged_data[star[i]], 'o', c=color(star)[i-1])
plt.gca().invert_yaxis()
plt.title('Observations of each star (averaged over all wavelengths)')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.show()
for i in range(1, 11, 1):
    plt.plot(X_mat[star[i]], averaged_data[star[i]], 'o', c=color(star)[i-1])
plt.gca().invert_yaxis()
plt.title('Observations of each star (averaged over all wavelengths)')
plt.xlabel('Airmass')
plt.ylabel('Magnitude')
plt.show()

# TODO automated quality control
# TODO passbands
# TODO output extinction corrected results
# TODO modular
# TODO seeing smoothing
# TODO residuals
