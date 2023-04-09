import numpy as np
import model
import scipy
from scipy import integrate
import cosmo
c_km = 299792.458 #km/s
c = c_km*1e3

def params(Survey1,Survey2,A_sky=None):

    # A_sky: manual input for overlapping sky area between two surveys. If None
    #   given, A_sky will default to the full area for the smallest survey,
    #   assuming 100% overlap.

    zmins,zmaxs = [],[] # Arrays to collect z limits for both surveys
    A_skys = []
    D_dish,t_tot,N_dish,nbar = None,None,None,None # Radio/Opt specific params. Set None as default

    ### Telescope/survey parameters:
    if Survey1=='MK_2019' or Survey2=='MK_2019': # MeerKAT 2019 pilot survey in L-band
        zmins.append(0.4);zmaxs.append(0.46)
        A_skys.append(100) # area in sq.deg
        t_tot = 5 # observation hours
        N_dish = 60 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
    if Survey1=='MK_LB' or Survey2=='MK_LB': # MeerKLASS UHF-band
        zmins.append(0.2);zmaxs.append(0.58)
        A_skys.append(4000) # area in sq.deg
        t_tot = 4000 # observation hours
        N_dish = 64 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
    if Survey1=='MK_UHF' or Survey2=='MK_UHF': # MeerKLASS UHF-band
        zmins.append(0.4);zmaxs.append(1.45)
        A_skys.append(10000) # area in sq.deg
        t_tot = 1250
        N_dish = 64 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
    if Survey1=='SKA' or Survey2=='SKA': # SKA Band 1
        zmins.append(0.35);zmaxs.append(3)
        A_skys.append(20000) # area in sq.deg
        t_tot = 10000 # observation hours
        N_dish = 197 # number of dishes
        D_dish = 15 # diameter of dish [metres]
    if Survey1=='DESI_LRG' or Survey2=='DESI_LRG': # DESI Luminous Red Galaxies
        zmins.append(0.4);zmaxs.append(1.1)
        A_skys.append(5000) # area in sq.deg
        ####### AMEND TO N(z) profile ##########
        nbar = 1e-4 # number density of galaxies
        ########################################
    elif Survey1=='DESI_ELG' or Survey2=='DESI_ELG': # DESI Emission Line Galaxies
        zmins.append(0.8);zmaxs.append(1.4)
        A_skys.append(5000) # area in sq.deg
        ####### AMEND TO N(z) profile ##########
        nbar = 1e-4 # number density of galaxies
        ########################################

    else: print('\n ERROR: Invalid survey selection: %s \n'%Survey); exit()
    ### Set cosmology:
    zmin,zmax = np.max(zmins),np.min(zmaxs) # overlapping redshift range for surveys
    z = np.mean([zmin,zmax]) # central redshift
    cosmo.SetCosmology(z=z)
    if D_dish is None: R_beam = None
    else: R_beam = BeamPars(D_dish,z)[1]
    A_sky = np.min(A_skys)
    V_bin = Vsur(zmin,zmax,A_sky)
    return z,zmin,zmax,R_beam,A_sky,t_tot,N_dish,nbar,V_bin

def params_gal(Survey):
    ### Telescope/survey parameters:
    if Survey=='DESI_LRG': # MeerKAT 2019 pilot survey in L-band
        zmin,zmax = 0.4,1.1
        A_sky = 5000 # area in sq.deg
        ####### AMEND TO N(z) profile ##########
        nbar = 1e-4 # number density of galaxies
        ########################################
    if Survey=='DESI_ELG': # MeerKAT 2019 pilot survey in L-band
        zmin,zmax = 0.8,1.4
        A_sky = 5000 # area in sq.deg
        ####### AMEND TO N(z) profile ##########
        nbar = 1e-4 # number density of galaxies
        ########################################
    z = np.mean([zmin,zmax]) # central redshift
    cosmo.SetCosmology(z=z)
    V_bin = Vsur(zmin,zmax,A_sky)
    return z,zmin,zmax,A_sky,nbar,V_bin

def BeamPars(D_dish,z):
    ''' Calclate FWHM of beam in deg and comoving [Mpc/h] '''
    d_c = cosmo.D_com(z) # Comoving distance to frequency bin
    nu = model.Red2Freq(z)
    theta_FWHM = np.degrees(c / (nu*1e6 * D_dish)) # Beam size deg
    sig_beam = theta_FWHM/(2*np.sqrt(2*np.log(2)))
    R_beam = d_c * np.radians(sig_beam) #Beam size [Mpc/h]
    return theta_FWHM,R_beam

def Vsur(zmin,zmax,Area):
    ''' Survey volume in [Mpc/h]^3 '''
    H0 = cosmo.H(0)
    Omega_A = Area * (np.pi/180)**2
    func = lambda z: Omega_A*c_km*cosmo.D_com(z)**2 / (H0*cosmo.E(z))
    return scipy.integrate.romberg(func,zmin,zmax)
