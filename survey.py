import numpy as np
import scipy
from scipy import integrate
import cosmo
c_km = 299792.458 #km/s

def params(Survey):

    if Survey=='MK_UHF':

        Area = 4000 # area in sq.deg
        zmin,zmax = 0.4,1.4

        ### Set cosmology:
        zc = (zmax - zmin)/2
        cosmo.SetCosmology(z=zc)

        V_sur = Vsur(zmin,zmax,Area)

    return Area,zmin,zmax,zc,V_sur


def Vsur(zmin,zmax,Area):
    ''' Survey volume in [Mpc/h]^3 '''
    H0 = cosmo.H(0)
    Omega_A = Area * (np.pi/180)**2
    func = lambda z: Omega_A*c_km*cosmo.D_com(z)**2 / (H0*cosmo.E(z))
    return scipy.integrate.romberg(func,zmin,zmax)
