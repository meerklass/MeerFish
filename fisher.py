import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy import linalg
from matplotlib.patches import Ellipse
import model
import cosmo

def V_eff(k,mu):
    return V_bin * ( model.P_HI(k,mu,z,Pmod,cosmopars,surveypars) / model.P_HI_obs(k,mu,z,Pmod,cosmopars,surveypars) )**2

def dlnP_dTbar(k,mu):
    return 2 / model.Tbar(z,Omega_HI)
def dlnP_dbHI(k,mu):
    return 2 / (b_HI + f*mu**2)
def dlnP_df(k,mu):
    return 2*mu**2 / (b_HI + f*mu**2)
def dlnP_dfNL(k,mu):
    return 2*bphiHI*M**(-1) / (b_HI + f*mu**2)

def dF(k,mu):
    return 1/(8*np.pi**2)*k**2*deriv_i(k,mu)*deriv_j(k,mu)*V_eff(k,mu)

def integrate2D(dfun,k,mu):
    #2D integration function
    muint = [scipy.integrate.simps(dfun.T[i], mu) for i in range(k.size)]
    return scipy.integrate.simps(muint, k)

def Matrix(theta,k,Pmod_,z_,cosmopars_,surveypars_,V_bin_):
    '''Compute Fisher matrix for parameter set theta'''

    global V_bin,z,Pmod,cosmopars,surveypars
    V_bin=V_bin_; z=z_; Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_

    global Omega_HI,b_HI,f,bphiHI,f_NL,M
    Omega_HI,b_HI,f,bphiHI,f_NL = cosmopars

    mu = np.linspace(-1,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)

    Npar = np.shape(theta)[1]
    F = np.zeros((Npar,Npar))
    global deriv_i; global deriv_j
    for i in range(Npar):
        def deriv_i(k_i,mu_i):
            #if theta[0,i]==r'$\overline{T}_{\rm HI}$': return dlnP_dTbar(k_i,mu_i)
            #if theta[0,i]==r'$b_{\rm HI}$': return dlnP_dbHI(k_i,mu_i)
            #if theta[0,i]==r'$f$': return dlnP_df(k_i,mu_i)
            if theta[0,i]=='bHI': return dlnP_dbHI(k_i,mu_i)
            if theta[0,i]=='f': return dlnP_df(k_i,mu_i)
            #if theta[0,i]==r'$f_{\rm NL}$': return dlnP_dfNL(k_i,mu_i)
        for j in range(Npar):
            if j>=i:
                def deriv_j(k_i,mu_i):
                    #if theta[0,j]==r'${T}_{\rm HI}$': return dlnP_dTbar(k_i,mu_i)
                    #if theta[0,j]==r'$b_{\rm HI}$': return dlnP_dbHI(k_i,mu_i)
                    #if theta[0,j]==r'$f$': return dlnP_df(k_i,mu_i)
                    if theta[0,j]=='bHI': return dlnP_dbHI(k_i,mu_i)
                    if theta[0,j]=='f': return dlnP_df(k_i,mu_i)
                    #if theta[0,j]==r'$f_{\rm NL}$': return dlnP_dfNL(k_i,mu_i)
                F[i,j] = integrate2D(dF(kgrid,mugrid),k,mu)
            else: F[i,j] = F[j,i]
    return F

def ContourEllipse(F,x,y):
    ''' Calculate ellipses using Eq2 and 4 from https://arxiv.org/pdf/0906.4123.pdf
           with inputs from Phil Bull's Fisher repo https://gitlab.com/radio-fisher/bao21cm/-/blob/master/radiofisher/baofisher.py'''
    # F: input Fisher matrix
    # (x,y): chosen indices for parameter pair in Fisher matrix to compute elipse for
    C = np.linalg.inv(F)
    Cxx,Cyy,Cxy = C[x,x],C[y,y],C[x,y]
    a = 2*np.sqrt( (Cxx + Cyy)/2 + np.sqrt( (Cxx - Cyy)**2/4 + Cxy**2 ) )
    b = 2*np.sqrt( (Cxx + Cyy)/2 - np.sqrt( (Cxx - Cyy)**2/4 + Cxy**2 ) )
    # Compute widths and heights, factors of two to agree with python Ellipse convention
    # Flip major/minor axis depending on which parameter is dominant
    if Cxx >= Cyy: w = 2*a; h = 2*b
    else: w = 2*b; h = 2*a
    ang = np.degrees( 0.5*np.arctan( 2*Cxy / (Cxx - Cyy) ) )
    return w,h,ang

def CornerPlot(F,theta):
    Npar = np.shape(F)[0]
    plt.figure(figsize=(10,10))
    for i in range(Npar):
        for j in range(Npar):
            plt.subplots(Npar,i+1,j+1)
            ax = plt.gca()
            w,h,ang = ContourEllipse(F,i,j)
            x,y = theta[1,i],theta[1,j]
            ellipse = Ellipse(xy=(x,y), width=w, height=h, angle=ang, edgecolor='r', fc='none', lw=2)
            ax.add_patch(ellipse)
            ax.autoscale()
            plt.axvline(b_HI,color='black',lw=1,ls='--')
            plt.axhline(f_NL,color='black',lw=1,ls='--')
            plt.xlabel(theta[0,i])
            plt.ylabel(theta[0,j])
    plt.show()
