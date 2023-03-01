import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import pylab as P


def dF(kk,mu):
    return (1./(8*pi*pi))*pow(kk,2)*deriv_i(kk,mu,zc)*deriv_j(kk,mu,zc)*Veff(kk,mu,zc)

#2D integration function
def integrate2D(dfun, kgrid, mugrid):

    muint = [scipy.integrate.simps(dfun.T[i], mugrid) for i in range(kgrid.size)]
    return scipy.integrate.simps(muint, kgrid)
mugrid = np.linspace(-1., 1., 200)


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
    y1 = 0.5*(c11 + c22)
    y2 = np.sqrt( 0.25*(c11 - c22)**2. + c12**2. )
    a = 2. * np.sqrt(y1 + y2) # Factor of 2 because def. is *total* width of ellipse
    b = 2. * np.sqrt(y1 - y2)

    # Flip major/minor axis depending on which parameter is dominant
    if c11 >= c22:
        w = a; h = b
    else:
        w = b; h = a

    # Handle c11==c22 case for angle calculation
    if c11 != c22:
        ang = 0.5*np.arctan( 2.*c12 / (c11 - c22) )
    else:
        ang = 0.5*np.arctan( 2.*c12 / 1e-20 ) # Sign sensitivity here

    return w, h, ang * 180./np.pi
