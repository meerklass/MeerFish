import numpy as np
import scipy
from scipy import integrate
from scipy import linalg
import model

def V_eff(k,mu):
    return V_bin * ( model.P_HI(k,mu,z,Pmod,cosmopars,surveypars) / model.P_HI_obs(k,mu,z,Pmod,cosmopars,surveypars) )**2

def dlnP_dbHI(k,mu):
    return 2 / (b_HI + f*mu**2)
def dlnP_df(k,mu):
    return 2*mu**2 / (b_HI + f*mu**2)

def dF(k,mu):
    return 1/(8*np.pi**2)*k**2*deriv_i(k,mu)*deriv_j(k,mu)*V_eff(k,mu)

def integrate2D(dfun,k,mu):
    #2D integration function
    muint = [scipy.integrate.simps(dfun.T[i], mu) for i in range(k.size)]
    return scipy.integrate.simps(muint, k)

def Matrix(theta,k,Pmod_,z_,cosmopars_,surveypars_,V_bin_):
    '''Compute Fisher matrix for parameter set theta'''

    global V_bin; global z; global Pmod; global cosmopars; global surveypars
    V_bin=V_bin_; z=z_; Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_

    global b_HI; global f
    Omega_HI,b_HI,f,bphiHI,f_NL = cosmopars


    mu = np.linspace(-1,1,200)
    kgrid,mugrid = np.meshgrid(k,mu)

    Npar = np.shape(theta)[0]
    F = np.zeros((Npar,Npar))
    global deriv_i; global deriv_j
    for i in range(Npar):
        def deriv_i(k_i,mu_i):
            if theta[0,i]=='bHI': return dlnP_dbHI(k_i,mu_i)
            if theta[0,i]=='f': return dlnP_df(k_i,mu_i)
        for j in range(Npar):
            if j>=i:
                def deriv_j(k_i,mu_i):
                    if theta[0,j]=='bHI': return dlnP_dbHI(k_i,mu_i)
                    if theta[0,j]=='f': return dlnP_df(k_i,mu_i)
                F[i,j] = integrate2D(dF(kgrid,mugrid),k,mu)
            else: F[i,j] = F[j,i]
    return F

def ellipse_for_fisher_params(p1, p2, F, Finv=None):
    """
    **********FROM PHIL BULL CODE: https://gitlab.com/radio-fisher/bao21cm/-/blob/master/radiofisher/baofisher.py

    Return covariance ellipse parameters (width, height, angle) from
    Fisher matrix F, for parameters in the matrix with indices p1 and p2.

    See arXiv:0906.4123 for expressions.
    """
    if Finv is not None:
        cov = Finv
    else:
        cov = np.linalg.inv(F)
    c11 = cov[p1,p1]
    c22 = cov[p2,p2]
    c12 = cov[p1,p2]

    # Calculate ellipse parameters (Eqs. 2-4 of Coe, arXiv:0906.4123)
    a = 2*np.sqrt( (c11 + c22)/2 + np.sqrt( (c11 - c22)**2/4 + c12**2 ) )
    b = 2*np.sqrt( (c11 + c22)/2 - np.sqrt( (c11 - c22)**2/4 + c12**2 ) )

    # Define widths and heights, factors of two to agree with python Ellipse convention
    # Flip major/minor axis depending on which parameter is dominant
    if c11 >= c22:
        w = 2*a; h = 2*b
    else:
        w = 2*b; h = 2*a

    # Handle c11==c22 case for angle calculation
    if c11 != c22:
        ang = 0.5*np.arctan( 2.*c12 / (c11 - c22) )
    else:
        ang = 0.5*np.arctan( 2.*c12 / 1e-20 ) # Sign sensitivity here

    #return w, h, ang * 180./np.pi
    return w,h,np.degrees(ang)
