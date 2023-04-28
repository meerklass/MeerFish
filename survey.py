import numpy as np
import model
import scipy
from scipy import integrate
import cosmo
import model
c_km = 299792.458 #km/s
c = c_km*1e3

def params(Survey1,Survey2=None,A_sky=None,zminzmax=None):

    # A_sky: manual input for overlapping sky area between two surveys. If None
    #   given, A_sky will default to the full area for the smallest survey,
    #   assuming 100% overlap.

    if Survey2==None: Survey2 = Survey1
    zmins,zmaxs = [],[] # Arrays to collect z limits for both surveys
    A_skys = []
    P_N,D_dish,t_tot,N_dish,nbar = None,None,None,None,None # Radio/Opt specific params. Set None as default

    ### Telescope/survey parameters:
    if Survey1=='MK_2019' or Survey2=='MK_2019': # MeerKAT 2019 pilot survey in L-band
        if zminzmax==None: zmins.append(0.4); zmaxs.append(0.46)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        z = np.mean([zmins[-1],zmaxs[-1]]) # initial central redshift
        cosmo.SetCosmology(z=z) # set initial cosmology for P_N
        A_skys.append(100) # area in sq.deg
        t_tot = 5 # observation hours
        N_dish = 60 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
        P_N = model.P_N(z,zmins[-1],zmaxs[-1],A_skys[-1],t_tot,N_dish)
    elif Survey1=='MK_LB' or Survey2=='MK_LB': # MeerKLASS UHF-band
        if zminzmax==None: zmins.append(0.2);zmaxs.append(0.58)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        z = np.mean([zmins[-1],zmaxs[-1]]) # initial central redshift
        cosmo.SetCosmology(z=z) # set initial cosmology for P_N
        A_skys.append(4000) # area in sq.deg
        t_tot = 4000 # observation hours
        N_dish = 64 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
        P_N = model.P_N(z,zmins[-1],zmaxs[-1],A_skys[-1],t_tot,N_dish)
    elif Survey1=='MK_UHF' or Survey2=='MK_UHF': # MeerKLASS UHF-band
        if zminzmax==None: zmins.append(0.4);zmaxs.append(1.45)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        z = np.mean([zmins[-1],zmaxs[-1]]) # initial central redshift
        cosmo.SetCosmology(z=z) # set initial cosmology for P_N

        A_skys.append(10000) # area in sq.deg
        t_tot = 1250

        #t_tot = 180
        #A_skys.append(6000)

        N_dish = 64 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
        P_N = model.P_N(z,zmins[-1],zmaxs[-1],A_skys[-1],t_tot,N_dish)
    elif Survey1=='SKA' or Survey2=='SKA': # SKA Band 1
        if zminzmax==None: zmins.append(0.35);zmaxs.append(3)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        z = np.mean([zmins[-1],zmaxs[-1]]) # initial central redshift
        cosmo.SetCosmology(z=z) # set initial cosmology for P_N
        A_skys.append(20000) # area in sq.deg
        t_tot = 10000 # observation hours
        N_dish = 197 # number of dishes
        D_dish = 15 # diameter of dish [metres]
        P_N = model.P_N(z,zmins[-1],zmaxs[-1],A_skys[-1],t_tot,N_dish)
    if Survey1=='DESI-LRG' or Survey2=='DESI-LRG': # DESI Luminous Red Galaxies
        if zminzmax==None: zmins.append(0.4);zmaxs.append(1.1)
        A_g = 5000
        A_skys.append(A_g) # area in sq.deg
        ####### AMEND TO N(z) profile ##########
        nbar = 1e-4 # number density of galaxies
        ########################################
    elif Survey1=='DESI-ELG' or Survey2=='DESI-ELG': # DESI Emission Line Galaxies
        if zminzmax==None: zmins.append(0.8);zmaxs.append(1.4)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        A_skys.append(5000) # area in sq.deg
        ####### AMEND TO N(z) profile ##########
        Ngal_per_deg2 = 2400 # from https://arxiv.org/pdf/2010.11281.pdf
        Ngal = Ngal_per_deg2 * A_skys[-1]
        nbar = Ngal / Vsur(zmins[-1],zmaxs[-1],A_skys[-1])
        ########################################
    elif Survey1==r'Euclid-H$\alpha$' or Survey2==r'Euclid-H$\alpha$': # Euclid spectro-z
        if zminzmax==None: zmins.append(0.9);zmaxs.append(1.8)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        A_skys.append(15000) # area in sq.deg
        ####### AMEND TO N(z) profile ##########
        nbar = 5.49e-4 # Mean of lowest 3 z-bins from Tab2: https://www.aanda.org/articles/aa/pdf/2020/10/aa38071-20.pdf
    elif Survey1=='4MOST' or Survey2=='4MOST': # 4MOST Fig2: http://www.eso.org/sci/publications/messenger/archive/no.175-mar19/messenger-no175-50-53.pdf
        if zminzmax==None: zmins.append(0);zmaxs.append(2)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        A_skys.append(7500) # area in sq.deg
        ####### AMEND TO N(z) profile ##########
        Ngal_per_deg2 = 1200 # from https://arxiv.org/pdf/2010.11281.pdf
        Ngal = Ngal_per_deg2 * A_skys[-1]
        nbar = Ngal / Vsur(zmins[-1],zmaxs[-1],A_skys[-1])
        ########################################

    #else: print('\n ERROR: Invalid survey selection\n'); exit()

    ### Set cosmology:
    zmin,zmax = np.max(zmins),np.min(zmaxs) # overlapping redshift range for surveys
    z = np.mean([zmin,zmax]) # central redshift
    if A_sky==None: A_sky = np.min(A_skys) # assumes 100% survey overlap
    V_bin = Vsur(zmin,zmax,A_sky)
    cosmo.SetCosmology(z=z)
    if D_dish is None: R_beam = None
    else: R_beam = BeamPars(D_dish,z)[1]
    return z,zmin,zmax,A_sky,V_bin,R_beam,t_tot,N_dish,P_N,nbar

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
    cosmo.SetCosmology(z=np.mean([zmin,zmax]))
    H0 = cosmo.H(0)
    Omega_A = Area * (np.pi/180)**2
    func = lambda z: Omega_A*c_km*cosmo.D_com(z)**2 / (H0*cosmo.E(z))
    return scipy.integrate.romberg(func,zmin,zmax)
