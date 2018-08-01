import time as runtime
start = runtime.time() # count runtime of program

import numpy as np
import matplotlib.pyplot as plt
import glob
import string

from scipy import interpolate, signal
from matplotlib.lines import Line2D

class Del:
  def __init__(self, keep=string.digits):
    self.comp = dict((ord(c),c) for c in keep)
  def __getitem__(self, k):
    return self.comp.get(k)

DD = Del()

import config as cfg # load in external configuration parameters

############################### DEFINING FUNTIONS AND PARAMETERS #############################################

version = '0.2b'

datadir = cfg.data_directory # data subdirectory
columndata = cfg.column_data # column data

wave1 = np.arange(5551.5, 10003.5, 7)
wave2 = np.arange(3001.8, 6001.3, 3.5)

def Magnitude(data): # convert photon counts to magnitude
    mag = -2.5*np.log10(data) + 15.0
    return mag

def invmag(m): # convert magnitude back to flux
    d = 10**((m-15.0)/(-2.5))
    return d

def ret(m_obs,k,X): # bouguer equation
    m_act = m_obs + k*X
    return m_act

def Trans(airmass,coeff):
    T = np.exp(-airmass * coeff)
    return T
    
def GaussConvolve(data):
    window = signal.gaussian(len(data),1)
    window = window/sum(window)
    conv = signal.convolve(data, window, mode='same')
    return conv

wave = np.sort(np.append(wave2,wave1))

microwave = wave/10000.

airmass = 1

#plt.figure(figsize=[12,6])
#plt.title('Extinction at X=1')
#plt.xlabel('Wavelength (Angstrom)')
#plt.ylabel('Magnitudes')

# =============================================================================
# RAYLEIGH SCATTERING
# =============================================================================

def rayleigh(wave,refr,h):
    A = (9.4977E-3) * ((1./wave)**4) * (refr**2) * (np.exp(-h/7.996))
    return A

def refractiveterm(wave):
    refr = (0.23465) + ((1.076E2/(146-(1./wave)**2))) + ((0.93161/(41-(1./wave)**2)))
    return refr

refr = refractiveterm(microwave)

A_ray = rayleigh(microwave, refr, 1.7) # mag/airmass

A_ray = A_ray * airmass # magnitudes

#plt.plot(wave,A_ray,c='red')

# =============================================================================
# DUST
# =============================================================================

def dust(A0, wave, alpha, h,H):
    A = A0 * wave**(-alpha) * np.exp(-h/H)
    return A

A0 = 0.03
alpha = 1.3
H = 1.5

A_dust = dust(A0, microwave, alpha, 1.7, H) # exp absorption

# convert to magnitudes

A_dust = A_dust * 1.08574

#plt.plot(wave,A_dust,c='green')

# =============================================================================
# OZONE
# =============================================================================

vacwaveoz, k_oz = np.loadtxt(columndata+'ozone.dat',unpack=True)

vacwaveoz = vacwaveoz * 10 # convert nm to angstroms
k_oz = k_oz * 2.687E19 # convert cm^2/mol to cm^-1, Loschmidt constant in mol/cm^3

ozwavenb=1E8/vacwaveoz # calculate wavenumber in cm^-1

# convert vacuum wavelength to air wavelength, taken from Kurucz' synthe
def airwave(vacwave,waven):
    airwave = vacwave / (1.0000834213 + 2406030.0 / (1.30E10 - waven**2) + 15997.0 / (3.89E9 - waven**2))
    return np.round(airwave,3)

airwaveoz = airwave(vacwaveoz, ozwavenb)

airwaveozc = GaussConvolve(airwaveoz)

def ozone(T,k):
    A = 1.11 * T * k
    return A

f_oz = interpolate.interp1d(airwaveozc,k_oz)

k_oz_interp = f_oz(wave)

A_oz = ozone(0.33, k_oz_interp)

#plt.plot(wave,A_oz,label='Ozone',c='orange')

# =============================================================================
# OXYGEN
# =============================================================================

oxygen = sorted(glob.glob(columndata+'oxygen/o2_*_hires_new')) # loads all relevant data files
oxcolumn = []

for i in oxygen:
    i = (str(i.translate(DD)))
    i = i[1:]
    oxcolumn.append(int(i))
    
oxcolumn = np.array(oxcolumn)

oxwave = []
k_ox = []

for file in oxygen:
    a, b = np.loadtxt(file,unpack='True')
    oxwave.append(a)
    k_ox.append(b)

oxwave = np.array(oxwave)
k_ox = np.array(k_ox)

oxwavenb = 1E8/oxwave

airwaveox = airwave(oxwave, oxwavenb)

waveox = np.broadcast_to(wave, [len(oxygen),len(wave)])

for i in range(len(oxygen)):
    
    airwaveox[i] = GaussConvolve(airwaveox[i])

    f_ox = interpolate.interp1d(airwaveox[i],k_ox[i],fill_value='extrapolate')
     
    k_ox_interp = f_ox(waveox)
    
    A_ox = 2 - np.log10(k_ox_interp*100)
    
#plt.plot(waveox[0], np.max(A_ox,axis=0),c='blue')
    
# =============================================================================
# WATER
# =============================================================================

water = sorted(glob.glob(columndata+'water/h2o_*_hires_new')) # loads all relevant data files
watcolumn = []

for i in water:
    i = (str(i.translate(DD)))
    i = i[1:]
    watcolumn.append(int(i))

watcolumn = np.array(watcolumn)

watwave = []
k_wat = []

for file in water:
    c, d = np.loadtxt(file,unpack='True')
    watwave.append(c)
    k_wat.append(d)

watwave = np.array(watwave)
k_wat = np.array(k_wat)

watwavenb = 1E8/watwave
airwavewat = airwave(watwave, watwavenb)

wavewat = np.broadcast_to(wave, [len(water),len(wave)])

for i in range(len(water)):

    airwavewat[i] = GaussConvolve(airwavewat[i])
    
    f_ox = interpolate.interp1d(airwavewat[i],k_wat[i],fill_value='extrapolate') 
     
    k_wat_interp = f_ox(wavewat)
    
    A_wat = 2 - np.log10(k_wat_interp*100)
    
#plt.plot(wavewat[0], np.max(A_wat,axis=0),c='darkblue')

custom_lines = [Line2D([0], [0], c='red'),
                Line2D([0], [0], c='green'),
                Line2D([0], [0], c='orange'),
                Line2D([0], [0], c='blue'),
                Line2D([0], [0], c='darkblue')]

###############################################################

A_wat = np.average(A_wat,axis=0)
A_ox = np.average(A_ox,axis=0)

A_tot = A_ray + A_dust + A_oz + A_ox + A_wat

#plt.xlim(min(wave),max(wave))
#plt.ylim(0,6)
#plt.legend(custom_lines, ['Rayleigh','Dust','Ozone', 'Oxygen', 'Water'],loc=0)
#plt.show()
#
#plt.figure(figsize=[12,6])
#plt.title('Extinction at X=1')
#plt.xlabel('Wavelength (Angstrom)')
#plt.ylabel('Magnitudes')
#plt.plot(wave,A_tot,'k',label='Total')
#plt.legend(loc=0)
#plt.xlim(min(wave),max(wave))
#plt.ylim(0,6)
#plt.show()

end = runtime.time()

print('----- ' + str(np.round(end - start,4)) + ' seconds -----')





