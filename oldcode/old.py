#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Nov 6 10:05:15 2017

@author: Ryan Brown
'''

############################################# LOADING EXTENSIONS #############################################

import time as runtime
start = runtime.time() # count runtime of program

import numpy as np
import matplotlib.pyplot as plt
import glob
#import collections
from astropy.io import fits
from scipy import optimize, interpolate
#from matplotlib.lines import Line2D

import config as cfg # load in external configuration parameters

np.seterr(all='ignore')

######################################### SETTING PARAMETERS ################################################

version = ''

directory = cfg.datadir # data subdirectory
order = str(cfg.order) # spectrograph order
columndata = cfg.columndir
fluxes = cfg.fluxdir
lower = int(cfg.lower) # load in wavelength cut off from config
upper = int(cfg.upper)
remove = set(cfg.remove) # to know which stars to remove
residthreshold = float(cfg.residthreshold) # load threshold values
countthreshold = float(cfg.countthreshold)

############################################## DEFINE FUNCTIONS ##############################################
def Magnitude(data): # convert photon counts to magnitude
    mag = -2.5*np.log10(data) + 15.0
    return mag

def invmag(m): # convert magnitude back to flux
    d = 10**((m-15.0)/(-2.5))
    return d

def ret(m_obs,k,X): # bouguer equation
    m_act = m_obs + k*X
    return m_act

def microwave(wave): # convert angstroms to microwaves
    microwave = wave/10000.
    return microwave

def rayleigh(wave,refr,h): # calculates Rayleigh scattering extinction
    A = (9.4977E-3) * ((1./wave)**4) * (refr**2) * (np.exp(-h/7.996))
    return A

def refractiveterm(wave): # calculates refractive term in Rayleigh equation
    refr = (0.23465) + ((1.076E2/(146-(1./wave)**2))) + ((0.93161/(41-(1./wave)**2)))
    return refr

def Dust(A0, wave, alpha, h,H): # calculates dust extinction
    A = A0 * wave**(-alpha) * np.exp(-h/H)
    return A

def ozone(T,k): # calculates ozone extinction
    A = 1.11 * T * k
    return A

def total(ray, dus, ozone): # sums extinction components
    A_tot = ray + dus + ozone
    return A_tot

################################################ LOADING IN DATA #############################################

print('\nWelcome to ASTRA',version)

# =============================================================================
#
# check = countthreshold + 1.0
# 
# while check > countthreshold: # checks occurance over residuals
#
# =============================================================================
    
files = sorted(glob.glob(directory+'*-'+order+'.fits')) # loads all relevant data files

filelist = [] # generate empty arrays to fill in for loop
starlist = []
data = []
wave = []
airmass = []
time = []
weights = []

print('\nLoading data...\n')

for file in files:
    print('\r'+str(file),end='',flush=True)
    hdulist = fits.open(file)
    filelist.append(file) # fills array with file names
    starlist.append(hdulist[0].header['OBJECT']) # fills array with star names
    airmass.append(hdulist[0].header['AIRMASS']) # fills array with airmass
    time.append(hdulist[0].header['SIDEREAL']) # fills array with sidereal time
    xrefval = hdulist[0].header['CRVAL1'] # minimum wavelength
    xrefpix = hdulist[0].header['CRPIX1'] # wavelength increment
    xinc = hdulist[0].header['CDELT1'] # spectrograph resolution
    raw_data = fits.getdata(file) # loads raw data
    data_tmp = raw_data / hdulist[0].header['EXPTIME'] # corrects for varying exposure times
    wavel = xrefval+(np.arange(len(data_tmp))+1-xrefpix)*xinc # creates array of wavelength used to truncate array 
    lowera = np.argwhere(wavel<lower) # calculate arguments where wavelength is less than threshold
    lowerb = np.max(lowera) # select max arg
    uppera = np.argwhere(wavel>upper) # calculate arguments where wavelength is more than threshold
    upperb = np.min(uppera) # select min arg
    data_tmp = data_tmp[lowerb:upperb] # truncate wavelength
    data.append(data_tmp) # fills array with corrected data
    wave.append(xrefval+(np.arange(len(data_tmp))+1-xrefpix)*xinc+(lowerb*xinc)) # fills array with wavelengths
    weights.append(raw_data[lowerb:upperb])
    hdulist.close()
    
wave = np.array(wave)
data = np.array(data)
weights = np.array(weights)

############################################ MASKING #################################################
 
#mask1 = np.logical_or(wave < 3800, wave > 4000)
#mask2 = np.logical_or(wave < 4060, wave > 4140)
#mask3 = np.logical_or(wave < 4300, wave > 4390)
#mask4 = np.logical_or(wave < 4810, wave > 4920)
#
#m1 = np.logical_and(mask1,mask2)
#m2 = np.logical_and(mask3,mask4)
#
#mask = np.logical_and(m1,m2)
#
#mask = np.invert(mask)
#
#data[mask] = 0
#wave[mask] = 0

########################################## PREPARING DATA ##############################################

stars = list(set(starlist)) # creates array of unique stars
stars = np.char.replace(stars,'Star','') # removes text string to convert to float
stars = set(np.asfarray(stars)) # converts to set

print('\n\nStars:',sorted(stars),'\n')

ext_stars = stars - set(remove) # set algebra to remove stars
ext_stars = np.asfarray(list(ext_stars)) # convert to array

stars = np.asfarray(list(stars)) # converts to array

starlist = np.char.replace(starlist,'Star','') 
starlist = np.asfarray(starlist)

################################### PLOT SAMPLE SPECTRA ################################################

#plt.figure(figsize=[10,5])
#
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

color = [n2c[number] for number in starlist]
#
#for i in range(1,275,1):
#
#    plt.plot(airmass[i],Magnitude(data[i,0]),'o',c=color[i])
#    
#plt.gca().invert_yaxis()
#plt.title('Bouguer Diagram')
#plt.xlabel('Airmass')
#plt.ylabel('Magnitude')
#plt.tight_layout()
#legend_elements = [Line2D([0], [0], marker='o', color='W', label='Star 1', markerfacecolor='C0', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 2', markerfacecolor='C1', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 3', markerfacecolor='C2', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 4', markerfacecolor='C3', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 5', markerfacecolor='C4', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 6', markerfacecolor='C5', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 7', markerfacecolor='C6', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 8', markerfacecolor='C7', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 9', markerfacecolor='C8', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 10', markerfacecolor='C9', markersize=10)]
# 
#plt.legend(handles=legend_elements,loc='lower right')
#plt.show()


#print('\nPlotting sample spectra...\n')
#
#samplestar = 1
#sam = np.min(np.argwhere(starlist==samplestar))
#
#plt.figure(figsize=[10,5])
#plt.plot(wave[sam],data[sam])
#plt.title('Sample Spectra Star ' + str(samplestar))
#plt.xlabel('Wavelength')
#plt.ylabel('Counts')
#plt.xlim(np.nanmin(wave[0]),np.nanmax(wave[0]))
#plt.tight_layout()
#plt.show()

############################################ PREPARE DATA ##############################################################

print('Removed Stars:',remove,'\n')

obs, nbwave = np.shape(wave) # extracts observation and wavelength data

print('Number of observations:',obs,'\n')

print('Wavelength Range:',np.nanmin(wave),'-',np.nanmax(wave),'Angstroms\n')

print('Wavelength Step:',xinc,'Angstroms\n')

print('Generating matrices...\n')

########################################### SETTING UP MATRICES ###############################################
 
a = np.zeros((len(ext_stars),len(starlist))) # collation index array
d = np.zeros((len(starlist),nbwave))   # data
w = weights # weights
X = np.zeros((len(starlist)))       # airmass
t = np.zeros((len(starlist)))   # time

for position1, item1 in enumerate(ext_stars):
        for position, item in enumerate(starlist):
            if item == item1:
                a[position1,position] = 1 # sets matrix element to 1 for corresponding stars
                d[position] = Magnitude(data[position]) # creates matrix of data
                X[position] = airmass[position] # creates matrix of airmass
                t[position] = time[position]-time[0] # creates matrix of time
                
############################################## PERFORMING MATRIX ALGEBRA ########################################
                
A = np.vstack([X,X*t,a]).T # design matrix
b = d/Magnitude(w) # weighting
alpha = np.dot(A.T,A)
beta = np.dot(A.T,b)
params = np.dot(np.linalg.inv(alpha),beta) # result matrix
cov = np.linalg.inv(alpha)
sigma = np.sqrt(np.sum((d-np.dot(A,params))**2))

########################################### PERFORMING LEAST SQUARES FITTING ######################################

def leastsq_function(params, *args): # define lst sq fitting function
    
    k = params[0] # reads corresponding column from result matrix
    kt = params[1]
    m = params[2:]
    X = args[0]
    t = args[1]
    y = args[2]
    a = args[3]
    yfit = k*X + kt*X*t + np.dot(m,a) ###################
    return y-yfit

m0 = np.zeros((len(ext_stars),nbwave)) # creates empty array of correct size
k0 = np.zeros((1,nbwave))
kt = np.zeros((1,nbwave))

params0 = np.concatenate((k0,kt,m0)) # empty array of parameters
resid0 = np.zeros((len(starlist),nbwave)) # empty array of residuals

print('Performing least squares fitting...\n')

for i in range(nbwave):
    result = optimize.leastsq(leastsq_function, params0[:,i], args=(X,t,d[:,i],a)) # optimize lst sq function using params0 as initial guess (all values zero)
    k0[0,i] = result[0][0]
    kt[0,i] = result[0][1]
    m0[:,i] = result[0][2:]
    resid0[:,i] = d[:,i] - (k0[0,i]*X + kt[0,i]*X*t + np.dot(m0[:,i],a)) # calculate residuals y - yfit

############################################# REMOVE DATA OUTSIDE THRESHOLD ######################################################

# =============================================================================
#
# indices = dict() # create empty dict to fill
# 
# for s in range(1,len(stars)+1,1):
#     
#     indices[s] = np.where(starlist == s)[0] # retrieves indices of all star X data
#     
# indices = list(indices.values()) # convert dict to list
#     
# removeindex = [] # empty array
# 
# for i in range(obs):
#     
#     if np.max(resid0[i]) > residthreshold or np.min(resid0[i]) < -residthreshold: # remove data if residual outside of threshold range
# #            resid0[i]=0.0
#         removeindex.append(i) # fill empty array
# 
# #    keep_row = resid0.any(axis=1)  # Index of rows with at least one non-zero value
# #    resid_non_zero = resid0[keep_row]  
#
#
#count = collections.Counter(starlist[removeindex]) # produces counts for number of readings for each star outside threshold
#   
#     print('\nOccurances of each star over residual threshold...\n')
#     
#     check = [] 
#     keys = []
#     
#     if len(count) == 0:
#         print('     No values lie outside residual threshold...')
#     else:
#         for key,value in count.items(): # prints star vs count
#             print(int(key),": ",value)
#             if value > countthreshold:
#                 check.append(value)
#                 keys.append(key)
#                 remove = remove ^ set(keys) # set addition
#                 
#                 print('Removing star',int(key))
#     
#     if len(check) != 0:
#         
#         check = np.max(check)
#         
#         print('\nRepeating with stars removed...')
#     
#     else:
#         
#         check = 0
#         
# print('\nValues lie within acceptable range, no more stars need removing')
#
# =============================================================================

################################################ MODELLING ############################################################
    
########### Rayleigh #############

microwave = microwave(wave)
refract = refractiveterm(microwave[0])
#ray = rayleigh(microwave[0], refract, 1.5) # wave, refractive, h

############## Dust ################

#dust = Dust(0.03, microwave[0], 1.3, 1.5, 1.5) # A0, wave, alpha, h, H

########### Ozone ####################

vacwaveoz, k_oz = np.loadtxt(columndata+'ozone.dat',unpack=True)
vacwaveoz = vacwaveoz * 10 # convert nm to angstroms
k_oz = k_oz * 2.687E19 # convert cm^2/mol to cm^-1, Loschmidt constant in mol/cm^3
ozwavenb=1E8/vacwaveoz # calculate wavenumber in cm^-1

# convert vacuum wavelength to air wavelength, taken from Kurucz' synthe
def airwave(vacwave,waven):
    airwave = vacwave / (1.0000834213 + 2406030.0 / (1.30E10 - waven**2) + 15997.0 / (3.89E9 - waven**2))
    return airwave

airwaveoz = airwave(vacwaveoz, ozwavenb)
interwave = interpolate.interp1d(np.round(airwaveoz,1),k_oz,fill_value='extrapolate')
inter = interwave(wave[0])
#ozone = ozone(0.22, inter) # T, wave

############ Total ############

#A_tot = total(ray, dust, ozone)

#################################################

refract = refractiveterm(microwave[0])

def fitfunc(wavel, h, A0, alpha, H, T):
    return ((9.4977E-3) * ((1./wavel)**4) * (refract**2) * (np.exp(-h/7.996))) + (A0 * wavel**(-alpha) * np.exp(-h/H)) + (1.11 * T * inter)

popt, pcov = optimize.curve_fit(fitfunc, microwave[0], k0[0],bounds=([0,0,0.5,0.5,0],[10,1,1.5,3,1]),p0=(1.5, 0.03, 1.3, 1.5, 0.33))

print('Fitted parameters:\n')

print('h = ' + str(popt[0])) 
print('A0 = ' + str(popt[1])) 
print('alpha = ' + str(popt[2])) 
print('H = ' + str(popt[3])) 
print('T = ' + str(popt[4]))

print('\nGenerating graphs...\n')

##################################### PLOTTING GRAPHS #################################################

########################## generate graphing colors #############################

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

color = [n2c[number] for number in starlist]

yfit = fitfunc(microwave[0],*popt)

plt.figure(figsize=[10,5])
plt.title('Extinction Coefficient with Residuals')
plt.plot(wave[0],k0[0],label='Bouguer Fit (Matrix)')
#plt.plot(wave[0],A_tot,label='Model with Assumed Parameters')
plt.plot(wave[0],yfit,lw=2,label='Model with Fitted Parameters')
plt.ylabel('Extinction Coefficient')
plt.xlim(np.nanmin(wave),np.nanmax(wave))
plt.ylim(0,1.2)
plt.legend(loc=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=[10,2.5])
plt.plot(wave[0],k0[0]-yfit)
plt.plot(wave[0],np.zeros(len(wave[0])),c='k',lw=1,zorder=1000) # zero line
plt.title('Fit Residual')
plt.xlabel('Wavelength (A)')
plt.ylabel('Extinction Coefficient')
plt.xlim(np.nanmin(wave),np.nanmax(wave))
plt.tight_layout()
plt.show()


#plt.figure(figsize=[10,5])
#for i in range(1,obs,1):
#    
#    plt.plot(wave[i],resid0[i],'o',c=color[i],alpha=0.75)
#    
#plt.plot(wave[0],np.zeros(len(wave[0])),c='k',lw=1,zorder=1000) # zero line
#plt.title('Star Removal Diagnostic')
#plt.xlabel('Wavelength (A)')
#plt.ylabel('Residuals')
#plt.xlim(np.nanmin(wave),np.nanmax(wave))
#
#
#legend_elements = [Line2D([0], [0], marker='o', color='W', label='Star 1', markerfacecolor='C0', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 2', markerfacecolor='C1', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 3', markerfacecolor='C2', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 4', markerfacecolor='C3', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 5', markerfacecolor='C4', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 6', markerfacecolor='C5', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 7', markerfacecolor='C6', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 8', markerfacecolor='C7', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 9', markerfacecolor='C8', markersize=10),
#                   Line2D([0], [0], marker='o', color='w', label='Star 10', markerfacecolor='C9', markersize=10)]
#
#leg = plt.legend(handles=legend_elements, loc='lower right')
#leg.set_zorder(10000)
#plt.tight_layout()
#plt.show()


plt.figure(figsize=[10,5])
plt.plot(wave[0],kt[0,:])
#plt.plot(wave[0],np.zeros(len(wave[0])),c='k',lw=1,zorder=1000) # zero line
plt.title('$k_t$')
plt.xlabel('Wavelength (A)')
plt.ylabel('$k_t$')
plt.xlim(np.nanmin(wave),np.nanmax(wave))
#plt.ylim(-0.004,0.004)
plt.tight_layout()
plt.show()

plt.figure(figsize=[10.6,6]) # 3d residual plot
plt.title('3d Residual Plot')
plt.imshow(resid0,clim=(-0.02,0.02), interpolation='none',extent=(np.nanmin(wave),np.nanmax(wave),0,len(starlist)), aspect='auto')
plt.xlabel('Wavelength')
plt.ylabel('Frame Number')
plt.colorbar()
plt.tight_layout()
plt.show()

fnwave, fnflux = np.loadtxt('data/star_fluxes/flux_Star01-2.dat', unpack=True)


print('\nFinished\n')

end = runtime.time()

print('----- ' + str(np.round(end - start,4)) + ' seconds -----')

######################## END OF CODE ###############################