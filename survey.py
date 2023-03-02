import numpy as np
import model
import scipy
from scipy import integrate
import cosmo
c_km = 299792.458 #km/s
c = c_km*1e3

def params(Survey):

    if Survey=='MK_UHF':
        ### Set cosmology:
        zmin,zmax = 0.4,1.4
        zc = (zmax - zmin)/2
        cosmo.SetCosmology(z=zc)
        ### Telescope/survey parameters:
        A_sky = 4000 # area in sq.deg
        t_tot = 4000 # observation hours
        N_dish = 64 # number of dishes
        D_dish = 13.5 # diameter of dish [metres]
        R_beam = BeamPars(D_dish,zc)[1]
        V_sur = Vsur(zmin,zmax,A_sky)
        
    return zc,zmin,zmax,R_beam,A_sky,t_tot,N_dish,V_sur

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
