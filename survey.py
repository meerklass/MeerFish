import numpy as np
import model
import scipy
from scipy import integrate
import cosmo
import model
c_km = 299792.458 #km/s
c = c_km*1e3

def params(Survey1='MK_UHF',Survey2=None,A_skyX=None,zminzmax=None,f_tobsloss=0,t_tot=0):

    # A_skyX: manual input for overlapping sky area between two surveys. If None
    #   given, A_sky will default to the full area for the smallest survey,
    #   assuming 100% overlap.

    # f_tobsloss: The fraction of IM obs time lost to flagging, failures etc.

    if Survey2==None: Survey2 = Survey1
    zmins,zmaxs = [],[] # Arrays to collect z limits for both surveys
    A_skys = []
    P_N,D_dish,N_dish,nbar = None,None,None,1e30 # Radio/Opt specific params. Set None/inf as default

    ### Telescope/survey parameters:
    if Survey1=='MK_2019' or Survey2=='MK_2019': # MeerKAT 2019 pilot survey in L-band
        if zminzmax is not None:
            zmin1,zmax1,zmin2,zmax2 = zminzmax[0],zminzmax[1],zminzmax[0],zminzmax[1]
            z =  np.mean([zminzmax[0],zminzmax[1]])
        else:
            if Survey1=='MK_2019':
                zmin1,zmax1 = 0.4,0.46
                z = np.mean([zmin1,zmax1])
            if Survey2=='MK_2019':
                zmin2,zmax2 = 0.4,0.46
                z = np.mean([zmin2,zmax2])
        cosmo.SetCosmology(z=z) # set initial cosmology for P_N
        A_sky1 = 100 # area in sq.deg
        if t_tot==0: t_tot = 5 * (1 - f_tobsloss) # observation hours
        N_dish = 60 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
        theta_FWHM,R_beam = BeamPars(D_dish,z)
        P_N = model.P_N(z,A_skys[-1],t_tot,N_dish,theta_FWHM=theta_FWHM)
    elif Survey1=='MK_LB' or Survey2=='MK_LB': # MeerKLASS UHF-band
        if zminzmax==None: zmins.append(0.2);zmaxs.append(0.58)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        z = np.mean([zmins[-1],zmaxs[-1]]) # initial central redshift
        cosmo.SetCosmology(z=z) # set initial cosmology for P_N
        A_skys.append(4000) # area in sq.deg
        if t_tot==0: t_tot = 4000 * (1 - f_tobsloss) # observation hours
        N_dish = 64 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
        theta_FWHM,R_beam = BeamPars(D_dish,z)
        P_N = model.P_N(z,A_skys[-1],t_tot,N_dish,theta_FWHM=theta_FWHM)
    elif Survey1=='MK_UHF' or Survey2=='MK_UHF': # MeerKLASS UHF-band
        if zminzmax==None: zmins.append(0.4);zmaxs.append(1.45)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        z = np.mean([zmins[-1],zmaxs[-1]]) # initial central redshift
        cosmo.SetCosmology(z=z) # set initial cosmology for P_N
        A_skys.append(10000) # area in sq.deg
        if t_tot==0: t_tot = 2500 * (1 - f_tobsloss)
        N_dish = 64 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
        theta_FWHM,R_beam = BeamPars(D_dish,z)
        P_N = model.P_N(z,A_skys[-1],t_tot,N_dish,theta_FWHM=theta_FWHM)
    elif Survey1=='SKA' or Survey2=='SKA': # SKA Band 1
        if zminzmax==None: zmins.append(0.35);zmaxs.append(3)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        z = np.mean([zmins[-1],zmaxs[-1]]) # initial central redshift
        cosmo.SetCosmology(z=z) # set initial cosmology for P_N
        A_skys.append(20000) # area in sq.deg
        if t_tot==0: t_tot = 10000 * (1 - f_tobsloss) # observation hours
        N_dish = 197 # number of dishes
        D_dish = 15 # diameter of dish [metres]
        theta_FWHM,R_beam = BeamPars(D_dish,z)
        P_N = model.P_N(z,A_skys[-1],t_tot,N_dish,theta_FWHM=theta_FWHM)
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
        A_skys.append(15000) # area in sq.deg
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
    elif Survey1=='Rubin' or Survey2=='Rubin':
        if zminzmax==None: zmins.append(0.3);zmaxs.append(3)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        A_skys.append(18000) # area in sq.deg
        Ngal_per_deg2 = 198000
        Ngal = Ngal_per_deg2 * A_skys[-1]
        nbar = Ngal / Vsur(zmins[-1],zmaxs[-1],A_skys[-1])
    elif Survey1=='Rubin_early' or Survey2=='Rubin_early': # as above just with half the number density and 10,000deg2
        if zminzmax==None: zmins.append(0.3);zmaxs.append(3)
        else: zmins.append(zminzmax[0]); zmaxs.append(zminzmax[1])
        A_skys.append(10000) # area in sq.deg
        Ngal_per_deg2 = 198000/2
        Ngal = Ngal_per_deg2 * A_skys[-1]
        nbar = Ngal / Vsur(zmins[-1],zmaxs[-1],A_skys[-1])
    ### Set cosmology:
    if len(zmins)==1: zmins.append(zmins[0]) # assume same zmin in second survey if none specified
    if len(zmaxs)==1: zmaxs.append(zmaxs[0]) # assume same zmax in second survey if none specified
    if len(A_skys)==1: A_skys.append(A_skys[0]) # assume same A_sky in second survey if none specified
    if A_skyX is None: A_skyX = np.min(A_skys)
    z = np.mean([np.max(zmins),np.min(zmaxs)]) # need to assume a single z_eff for all surveys
    V_bin1 = Vsur(zmins[0],zmaxs[0],A_skys[0])
    V_bin2 = Vsur(zmins[1],zmaxs[1],A_skys[1])
    V_binX = Vsur(np.max(zmins),np.min(zmaxs),A_skyX)
    if D_dish is None: theta_FWHM = None
    sigma_z1,sigma_z2 = 0,0 # Currently assuming no redshift error
    theta_FWHM2 = 0 # always assume no beam in survey 2 (i.e. galaxy survey)
    return z,zmins[0],zmins[1],zmaxs[0],zmaxs[1],A_skys[0],A_skys[1],V_bin1,V_bin2,V_binX,theta_FWHM,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar

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
