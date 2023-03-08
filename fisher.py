import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy import linalg
import scipy.stats as stats
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
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
    return 2*bphiHI*cosmo.M(k,z)**(-1) / (b_HI + f*mu**2)

def Matrix_2D(theta_ids,k,Pmod_,z_,cosmopars_,surveypars_,V_bin_):
    '''Compute full 2D anisotroic Fisher matrix for parameter set [theta]'''

    global V_bin,z,Pmod,cosmopars,surveypars
    V_bin=V_bin_; z=z_; Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_

    global Omega_HI,b_HI,f,bphiHI,f_NL,M
    Omega_HI,b_HI,f,bphiHI,f_NL = cosmopars

    mu = np.linspace(-1,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)

    Npar = len(theta_ids)
    F = np.zeros((Npar,Npar))
    global deriv_i; global deriv_j
    for i in range(Npar):
        def deriv_i(k_i,mu_i):
            if theta_ids[i]==r'$\overline{T}_{\rm HI}$': return dlnP_dTbar(k_i,mu_i)
            if theta_ids[i]==r'$b_{\rm HI}$': return dlnP_dbHI(k_i,mu_i)
            if theta_ids[i]==r'$f$': return dlnP_df(k_i,mu_i)
            if theta_ids[i]==r'$f_{\rm NL}$': return dlnP_dfNL(k_i,mu_i)
        for j in range(Npar):
            if j>=i: # avoid calculating symmetric off-diagonals twice
                def deriv_j(k_i,mu_i):
                    if theta_ids[j]==r'$\overline{T}_{\rm HI}$': return dlnP_dTbar(k_i,mu_i)
                    if theta_ids[j]==r'$b_{\rm HI}$': return dlnP_dbHI(k_i,mu_i)
                    if theta_ids[j]==r'$f$': return dlnP_df(k_i,mu_i)
                    if theta_ids[j]==r'$f_{\rm NL}$': return dlnP_dfNL(k_i,mu_i)

                dFkmu = 1/(8*np.pi**2)*kgrid**2*deriv_i(kgrid,mugrid)*deriv_j(kgrid,mugrid)*V_eff(kgrid,mugrid)
                dFk = [scipy.integrate.simps(dFkmu.T[i], mu) for i in range(k.size)] # integrate over mu
                F[i,j] = scipy.integrate.simps(dFk, k) # integrate over k
            else: F[i,j] = F[j,i]
    return F

def Matrix_ell(theta_ids,k,Pmod_,z_,cosmopars_,surveypars_,V_bin_,ell=0):
    '''Compute Fisher matrix for multipoles with parameter set [theta]'''

    global V_bin,z,Pmod,cosmopars,surveypars
    V_bin=V_bin_; z=z_; Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_

    global Omega_HI,b_HI,f,bphiHI,f_NL,M
    Omega_HI,b_HI,f,bphiHI,f_NL = cosmopars

    mu = np.linspace(-1,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)

    Npar = len(theta_ids)
    F = np.zeros((Npar,Npar))
    global deriv_i; global deriv_j
    for i in range(Npar):
        def deriv_i(k_i,mu_i):
            if theta_ids[i]==r'$\overline{T}_{\rm HI}$': return dlnP_dTbar(k_i,mu_i)
            if theta_ids[i]==r'$b_{\rm HI}$': return dlnP_dbHI(k_i,mu_i)
            if theta_ids[i]==r'$f$': return dlnP_df(k_i,mu_i)
            if theta_ids[i]==r'$f_{\rm NL}$': return dlnP_dfNL(k_i,mu_i)
        for j in range(Npar):
            if j>=i: # avoid calculating symmetric off-diagonals twice
                def deriv_j(k_i,mu_i):
                    if theta_ids[j]==r'$\overline{T}_{\rm HI}$': return dlnP_dTbar(k_i,mu_i)
                    if theta_ids[j]==r'$b_{\rm HI}$': return dlnP_dbHI(k_i,mu_i)
                    if theta_ids[j]==r'$f$': return dlnP_df(k_i,mu_i)
                    if theta_ids[j]==r'$f_{\rm NL}$': return dlnP_dfNL(k_i,mu_i)

                dFkmu = 1/(8*np.pi**2)*kgrid**2*deriv_i(kgrid,mugrid)*deriv_j(kgrid,mugrid)*V_eff(kgrid,mugrid,ell)
                dFk = [scipy.integrate.simps(dFkmu.T[i], mu) for i in range(k.size)] # integrate over mu
                F[i,j] = scipy.integrate.simps(dFk, k) # integrate over k
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

def CornerPlot(F,theta,theta_labels):
    Npar = np.shape(F)[0]
    C = np.linalg.inv(F)
    fig = plt.figure(figsize=(8,8))
    gs = GridSpec(Npar,Npar) # rows,columns
    for i in range(Npar):
        for j in range(Npar):
            if j>i: continue
            ax = fig.add_subplot(gs[i,j]) # First row, first column
            if i==(Npar-1): ax.set_xlabel(theta_labels[j])
            if j==0: ax.set_ylabel(theta_labels[i])
            if i==j: # Plot Gaussian distribution for marginalised parameter estimate
                sigma = np.sqrt(C[i,i])
                gauss_dummy = np.linspace(theta[i]-4*sigma, theta[i]+4*sigma, 100)
                ax.plot(gauss_dummy, stats.norm.pdf(gauss_dummy, theta[i], sigma),color='tab:blue')
                ax.set_ylim(bottom=0)
                ax.set_yticks([])
                if theta[i]==0: title = theta_labels[i]+r'$=0\pm%s$'%np.round(sigma,3)
                else: title = r'$\sigma($'+theta_labels[i]+r'$)/$'+theta_labels[i]+r'$=%s$'%(np.round(100*sigma/theta[i],3))+'%'
                ax.set_title(title)
                continue
            w,h,ang = ContourEllipse(F,j,i) # (j,i) reversed because panelling lower corner
            ellipse = Ellipse(xy=(theta[j],theta[i]), width=w, height=h, angle=ang, edgecolor='tab:blue', fc='none', lw=2)
            ax.add_patch(ellipse)
            ax.autoscale()
    return
