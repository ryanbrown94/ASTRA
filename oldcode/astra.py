#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:25:15 2018

@author: Ryan Brown
"""

# LOADING EXTENSIONS

import time as runtime
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from astropy.io import fits
from scipy import optimize, interpolate

import config as cfg  # load in external configuration parameters
import ext_modelling as ext  # load in model from external file

start = runtime.time()  # count runtime of program

np.seterr(all='ignore')

# SETTING PARAMETERS

version = '1.0'

directory = cfg.datadir  # data subdirectories
columndata = cfg.columndir
fluxes = cfg.fluxdir
order = str(cfg.order)  # spectrograph order
lower = int(cfg.lower)  # load in wavelength cut offs from config
upper = int(cfg.upper)
remove = set(cfg.remove)  # to know which stars to remove

# DEFINE FUNCTIONS


def magnitude(counts):  # convert photon counts to magnitude
    mag = -2.5*np.log10(counts) + 15.0
    return mag


def invmag(m):  # convert magnitude back to flux
    inv = 10**((m-15.0)/(-2.5))
    return inv


def bouguer(m_obs, k, x):  # bouguer equation
    m_act = m_obs + k*x
    return m_act


def microwave(wavelength):  # convert angstroms to microwaves
    microwavelength = wavelength/10000.
    return microwavelength


def rayleigh(wave, refr, h):  # calculates Rayleigh scattering extinction
    A = (9.4977E-3) * ((1./wave)**4) * (refr**2) * (np.exp(-h/7.996))
    return A


def refractiveterm(wave):  # calculates refractive term in Rayleigh equation
    refr = (0.23465) + ((1.076E2/(146-(1./wave)**2))) + \
    ((0.93161/(41-(1./wave)**2)))
    return refr


def dust(A0, wave, alpha, h):  # calculates dust extinction
    A = A0 * wave**(-alpha) * np.exp(-h/1.5)
    return A


def ozone(T,k):  # calculates ozone extinction
    A = 1.11 * T * k
    return A


def total(ray, dus, ozone):  # sums extinction components
    A_tot = ray + dus + ozone
    return A_tot


# convert vacuum wavelength to air wavelength, taken from Kurucz' synthe
def airwave(vacwave,waven):
    airwave = vacwave / (1.0000834213 + 2406030.0 / (1.30E10 - waven**2) + \
                         15997.0 / (3.89E9 - waven**2))
    return airwave


def color(starlist):  # assign each star a unique color
    
    n2c = {1: 'C0', # dark blue
           2: 'C1', # orange
           3: 'C2', # green
           4: 'C3', # red
           5: 'C4', # purple
           6: 'C5', # brown
           7: 'C6', # pink
           8: 'C7', # gray
           9: 'C8', # yellow
           10: 'C9'} # light blue
    
    c = [n2c[number] for number in starlist]
    
    return c


# LOADING IN DATA

print('\nWelcome to ASTRA',version)

files = sorted(glob.glob(directory+'*-'+order+'.fits'))  # loads all
                                                         # relevant data files

filelist = []  # generate empty arrays to fill in for loop
starlist = []
data = []
wave = []
airmass = []
time = []
weights = []

print('\nLoading data...\n')

for file in files:
    print(str(file)+'\n',end='',flush=True)  # prints each file name as it loads
    hdulist = fits.open(file)  # opens *.fits file
    filelist.append(file)  # fills array with file names
    starlist.append(hdulist[0].header['OBJECT'])  # fills array with star names
    airmass.append(hdulist[0].header['AIRMASS'])  # fills array with airmass
    time.append(hdulist[0].header['SIDEREAL'])  # fills array with sidereal time
    xrefval = hdulist[0].header['CRVAL1']  # minimum wavelength
    xrefpix = hdulist[0].header['CRPIX1']  # reference pixel coordinates
    xinc = hdulist[0].header['CDELT1']  # spectrograph resolution
    raw_data = fits.getdata(file)  # loads raw data for manipulation
    data_tmp = raw_data / hdulist[0].header['EXPTIME']  # corrects for varying
                                                        #  exposure times
    wavel = xrefval+(np.arange(len(data_tmp))+1-xrefpix)*xinc  # creates array
                                                               # of wavelength
    
    lowera = np.argwhere(wavel<lower)  # calculate arguments where wavelength
                                       # is less than threshold
    
    if not lowera.any():  # allows use of all values in array
        lowerb = np.argmin(wavel)
    else:
        lowerb = np.max(lowera)  # select max argument
        
    uppera = np.argwhere(wavel>upper)  # calculate arguments where wavelength
                                       # is more than threshold
    
    if not uppera.any():  # allows use of all values in array
        upperb = np.argmax(wavel)
    else:
        upperb = np.min(uppera)  # select min argument
        
    data_tmp = data_tmp[lowerb:upperb]  # truncate wavelength array if necesary
    data.append(data_tmp)  # fills data array with corrected data
    wave.append(xrefval+(np.arange(len(data_tmp))+1-xrefpix)*xinc+ \
                (lowerb*xinc))  # fills array with corrected wavelengths
    weights.append(raw_data[lowerb:upperb])  # fills array with raw
                                             # data for weighting
    hdulist.close()
    
wave = np.array(wave)  # convert to numpy arrays
data = np.array(data)
weights = np.array(weights)

print('\nDone!')

# MASKING

mask1 = np.logical_or(wave < 4090, wave > 4140)  # to remove balmer lines
mask2 = np.logical_or(wave < 4340, wave > 4390)
mask3 = np.logical_or(wave < 4860, wave > 4910)

mask = np.logical_and.reduce((mask1,mask2, mask3))
mask = np.invert(mask)

# PREPARING DATA

stars = list(set(starlist))  # creates array of unique stars
stars = np.char.replace(stars,'Star','')  # removes text string to convert
                                          # to float
stars = set(np.asfarray(stars))  # converts to set
ext_stars = stars - set(remove)  # set algebra to remove stars
ext_stars = np.asfarray(list(ext_stars))  # convert to array
stars = np.asfarray(list(stars))  # converts to array

print('\nStars:',np.round(stars,0),'\n')

starlist = np.char.replace(starlist,'Star','')  # array showing which star
                                                # corresponds to observation
starlist = np.asfarray(starlist)

if len(remove) == 0:
    print('No stars removed\n')
else:
    print('Removed Stars:',remove,'\n')

obs, nbwave = np.shape(wave)  # extracts observation and wavelength data

star = {}

for i in stars:
    star[i] = np.argwhere(starlist == i) 

print('Number of observations:',obs,'\n')
print('Wavelength Range:',np.nanmin(wave),'-',np.nanmax(wave),'Angstroms\n')
print('Wavelength Step:',xinc,'Angstroms\n')
print('Plotting Bouguer Diagram')

# PLOT BOUGUER
    
plt.figure(figsize=[10,5])
col = color(starlist)

for i in range(1,275,1):

    plt.plot(airmass[i],magnitude(data[i,0]),'o',c=col[i])
    
plt.gca().invert_yaxis()
plt.title('Bouguer Diagram')
plt.xlabel('Airmass')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# SETTING UP MATRICES

print('\nGenerating matrices...\n')

a = np.zeros((len(ext_stars),len(starlist)))  # collation index array
d = np.zeros((len(starlist),nbwave))  # data
X = np.zeros((len(starlist)))  # airmass
t = np.zeros((len(starlist)))  # time

for position1, item1 in enumerate(ext_stars):
        for position, item in enumerate(starlist):
            if item == item1:
                a[position1,position] = 1  # sets matrix element to 1 for
                                           # corresponding stars
                d[position] = magnitude(data[position])  # creates matrix
                                                         # of data
                X[position] = airmass[position]  # creates matrix of airmass
                t[position] = time[position]-time[0]  # creates matrix of time
                
d[mask] = 0
                
# LEAST SQUARES FITTING
                
# PERFORMING MATRIX ALGEBRA

A = np.vstack([X,X*t,a]).T # design matrix
b = d/magnitude(weights)
alpha = np.dot(A.T,A)
beta = np.dot(A.T,b)
cov = np.linalg.inv(alpha)
params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
sigma = np.sqrt(np.sum((d-np.dot(A,params))**2)/(len(t)-3))

########## ALTERNATIVE METHOD ##########

def leastsq_function(parameters, *args):  # define lst sq fitting function
    
    k = parameters[0]  # reads corresponding column from result matrix
    kt = parameters[1]
    m = parameters[2:]
    X = args[0]
    t = args[1]
    y = args[2]
    a = args[3]
    yfit = k*X + kt*X*t + np.dot(m,a)
    return y-yfit

m0 = np.zeros((len(ext_stars),nbwave))  # creates empty array of correct size
k0 = np.zeros((1,nbwave))
kt = np.zeros((1,nbwave))

resid0 = np.zeros((len(starlist),nbwave))  # empty array of residuals

print('Performing least squares fitting...\n')

for i in range(nbwave):
    result = optimize.leastsq(leastsq_function, params[:,i], \
                              args=(X,t,d[:,i],a))
    # optimize lst sq function using params as initial guess (from matrix)
    k0[0,i] = result[0][0]
    kt[0,i] = result[0][1]
    m0[:,i] = result[0][2:]
    resid0[:,i] = d[:,i] - (k0[0,i]*X + kt[0,i]*X*t + np.dot(m0[:,i],a)) 
    # calculate residuals y - yfit

params[0][params[0]==0] = np.nan  # allows interpolation to remove balmer lines
params[0] = pd.DataFrame(params[0]).interpolate().values.ravel().tolist()
    
# FITTED MODEL
    
vacwaveoz, k_oz = np.loadtxt(columndata+'ozone.dat',unpack=True) 
# load in ozone file
vacwaveoz = vacwaveoz * 10  # convert nm to angstroms
k_oz = k_oz * 2.687E19  # convert cm^2/mol to cm^-1, Loschmidt
                        # constant in mol/cm^3
ozwavenb=1E8/vacwaveoz  # calculate wavenumber in cm^-1
airwaveoz = airwave(vacwaveoz, ozwavenb)  # convert wavelength
interwave = interpolate.interp1d(np.round(airwaveoz,1),k_oz, \
                        fill_value='extrapolate')  # interpolate wavelength
                                                   # to correct scale
inter = interwave(wave[0])

microwave = microwave(wave)  # convert wavelength to microwave
refract = refractiveterm(microwave[0])  # calculate rayleigh refractive term

h = 1.7  # km ASTRA observing height

ray = ((9.4977E-3) * ((1./microwave[0])**4) * (refract**2) * (np.exp(-h/7.996)))
ray = np.tile(ray, (275,1))
ray = np.ndarray.flatten(ray)
oz = (1.11 * 0.33 * inter)
oz = np.tile(oz, (275,1))
oz = np.ndarray.flatten(oz)
A_wat = ext.A_wat
wat = A_wat[lowerb:upperb]
wat = np.tile(wat, (275,1))
wat = np.ndarray.flatten(wat)

def fitfunc(wavel, wat_multi, oz_multi, ray_multi, A0, alp):
    return ray_multi*ray + \
           oz_multi* oz + \
           wat_multi*wat + \
           (A0*wavel**(-alp) * np.exp(-h/1.5))
           
popt = {}
pcov = {}

fitparams = np.tile(params[0], (275,1))
fitparams = np.ndarray.flatten(fitparams)

fitwave = np.ndarray.flatten(microwave)

popt, pcov = optimize.curve_fit(fitfunc, fitwave,\
                              fitparams,p0=[1, 1, 1, 0.03, 1.3],\
                               bounds=((0.99, 0.99, 0.99, 0.01, 0.5),\
                                       (1, 1, 1, 0.05, 2.5)))

#print('Fitted parameters:\n')
#print('Water Multiplier = ' + str(popt[0])) 
#print('Ozone Multiplier = ' + str(popt[1])) 
#print('Rayleigh Multiplier = ' + str(popt[2])) 
#print('A0 = ' + str(popt[3]))
#print('Alpha = ' + str(popt[4]))

print('\nGenerating graphs...\n')

# PLOTTING GRAPHS

modfit = fitfunc(fitwave[0],*popt)  # fitted model using calculated parameters
modfit = np.reshape(modfit,(275,786))
extmodel = ext.A_tot  # load in external model
extwave = ext.wave

modelwave, modeldata = np.loadtxt('data/extdata/data0001-'+ order + '.ext',\
                                  unpack=True,skiprows=27)

plt.figure(figsize=[10,5])
plt.title('Extinction Coefficient')
plt.plot(wave[0],params[0],label='Least Squares Fitting',c='C0')
plt.plot(wave[0],modfit[0],lw=2,label='Model with Fitted Parameters',c='C1')
plt.plot(extwave,extmodel,label='External Model',c='C2')
plt.plot(modelwave,(-2.5*np.log10(modeldata)/2.3031),label='Barry Model',c='C3')
plt.axvline(x=4100,label='Balmer Lines',c='k',ls='--',lw=0.75)
plt.axvline(x=4340,c='k',ls='--',lw=0.75)
plt.axvline(x=4860,c='k',ls='--',lw=0.75)
plt.xlabel('Wavelength (A)')
plt.ylabel('$k_0$')
plt.xlim(np.nanmin(wave),np.nanmax(wave))
plt.ylim(0,1.2)
plt.legend(loc=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=[10,10])
plt.subplot(4,1,1)
plt.plot(wave[0],params[0] - extmodel[lowerb:upperb])
plt.axhline(y=0,c='k')
plt.title('Difference Between Data and Ryan Model')
plt.ylabel('$\Delta k$')
plt.xlim(np.nanmin(wave),np.nanmax(wave))

plt.subplot(4,1,2)
plt.plot(wave[0],params[0] - -2.5*np.log10(modeldata[lowerb:upperb])/2.330201)
plt.axhline(y=0,c='k')
plt.title('Difference Between Data and Barry Model')
plt.ylabel('$\Delta k$')
plt.xlim(np.nanmin(wave),np.nanmax(wave))

plt.subplot(4,1,3)
plt.plot(wave[0],params[0] - modfit[0])
plt.axhline(y=0,c='k')
plt.title('Difference Between Data and Fitted Model')
plt.xlabel('Wavelength (A)')
plt.ylabel('$\Delta k$')
plt.xlim(np.nanmin(wave),np.nanmax(wave))

plt.subplot(4,1,4)
plt.plot(wave[0],extmodel[lowerb:upperb] - \
         -2.5*np.log10(modeldata[lowerb:upperb])/2.330201)
plt.axhline(y=0,c='k')
plt.title('Difference Between Ryan Model and Barry Model')
plt.xlabel('Wavelength (A)')
plt.ylabel('$\Delta k$')
plt.xlim(np.nanmin(wave),np.nanmax(wave))
plt.tight_layout()
plt.show()

plt.figure(figsize=[10,5])
plt.imshow(resid0,clim=(-0.04,0.04),aspect='auto')
plt.tight_layout()
plt.show()

#plt.figure(figsize=[10,5])
#plt.plot(wave[0],params[1,:])
#plt.title('Time Component of Extinction')
#plt.xlabel('Wavelength (A)')
#plt.ylabel('$k_t$')
#plt.xlim(np.nanmin(wave),np.nanmax(wave))
#plt.tight_layout()
#plt.show()

#plt.figure(figsize=[10,5])
#col = color(starlist)
#
#for i in range(1,10,1):
#
#    plt.plot(X[star[i]],d[star[i],0],'o')
#    
#plt.gca().invert_yaxis()
#plt.title('Bouguer Diagram')
#plt.xlabel('Time')
#plt.ylabel('Magnitude')
#plt.tight_layout()
#plt.show()

print('\nFinished\n')

end = runtime.time()

print('----- ' + str(np.round(end - start,4)) + ' seconds -----')

# END OF CODE