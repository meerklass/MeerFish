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

def dPell_dtheta(ell,k,derivfunc):
    '''Generic derivitive multipole function, specify ln derivtive parameter model with derivfunc
        e.g. derivfunc = dlnP_dbHI for b_HI parameter'''
    integrand = lambda mu: derivfunc(k_i,mu) * model.P_HI(k_i,mu,z,Pmod,cosmopars,surveypars) * model.L(ell,mu)
    res = np.zeros(len(k))
    for i in range(len(k)):
        k_i = k[i]
        res[i] = (2*ell+1) * scipy.integrate.quad(integrand, 0, 1)[0]
    return res

def Matrix_2D(theta_ids,k,Pmod_,z_,cosmopars_,surveypars_,V_bin_):
    '''Compute full 2D anisotroic Fisher matrix for parameter set [theta]'''

    global V_bin,z,Pmod,cosmopars,surveypars
    V_bin=V_bin_; z=z_; Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_

    global Omega_HI,b_HI,f,bphiHI,f_NL,M
    Omega_HI,b_HI,f,bphiHI,f_NL = cosmopars

    mu = np.linspace(0,1,1000)
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

                dFkmu = kgrid**2*deriv_i(kgrid,mugrid)*deriv_j(kgrid,mugrid)*V_eff(kgrid,mugrid)
                dFk = [scipy.integrate.simps(dFkmu.T[i], mu) for i in range(k.size)] # integrate over mu
                F[i,j] = 1/(4*np.pi**2)*scipy.integrate.simps(dFk, k) # integrate over k
            else: F[i,j] = F[j,i]
    return F

def Matrix_ell(theta_ids,k,Pmod_,z_,cosmopars_,surveypars_,V_bin_,ells=[0,2,4]):
    '''Compute Fisher matrix for multipoles with parameter set [theta]'''

    global V_bin,z,Pmod,cosmopars,surveypars
    V_bin=V_bin_; z=z_; Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_

    global Omega_HI,b_HI,f,bphiHI,f_NL,M
    Omega_HI,b_HI,f,bphiHI,f_NL = cosmopars

    Npar = len(theta_ids)
    F = np.zeros((Npar,Npar))
    global deriv_i; global deriv_j

    for i in range(Npar):
        def deriv_i(ell,k):
            if theta_ids[i]==r'$\overline{T}_{\rm HI}$': return dPell_dtheta(ell,k,dlnP_dTbar)
            if theta_ids[i]==r'$b_{\rm HI}$': return dPell_dtheta(ell,k,dlnP_dbHI)
            if theta_ids[i]==r'$f$': return dPell_dtheta(ell,k,dlnP_df)
            if theta_ids[i]==r'$f_{\rm NL}$': return dPell_dtheta(ell,k,dlnP_dfNL)
        for j in range(Npar):
            if j>=i: # avoid calculating symmetric off-diagonals twice
                def deriv_j(ell,k):
                    if theta_ids[j]==r'$\overline{T}_{\rm HI}$': return dPell_dtheta(ell,k,dlnP_dTbar)
                    if theta_ids[j]==r'$b_{\rm HI}$': return dPell_dtheta(ell,k,dlnP_dbHI)
                    if theta_ids[j]==r'$f$': return dPell_dtheta(ell,k,dlnP_df)
                    if theta_ids[j]==r'$f_{\rm NL}$': return dPell_dtheta(ell,k,dlnP_dfNL)
                for ell0 in ells:
                    for ell1 in ells:
                        #if ell1<ell0: continue # avoid double counting multipole permutations e.g. ell=0,ell=1 == ell=1,ell=0
                        print(theta_ids[i],theta_ids[j],ell0,ell1)
                        dFk = k**2 * deriv_i(ell0,k) * Cov_ell(ell0,ell1,k)**(-1) * deriv_j(ell1,k)
                        F[i,j] += scipy.integrate.simps(dFk, k) # integrate over k and sum all mutlipole permutations
            else: F[i,j] = F[j,i]
    F *= V_bin/(4*np.pi**2)
    return F

def Cov_ell(ell0,ell1,k):
    ''' Covariance matrix for multipoles '''
    integrand = lambda mu: model.L(ell0,mu)*model.L(ell1,mu) * model.P_HI_obs(k_i,mu,z,Pmod,cosmopars,surveypars)**2
    C = np.zeros(len(k))
    for i in range(len(k)):
        k_i = k[i]
        C[i] = (2*ell0+1)*(2*ell1+1) * scipy.integrate.quad(integrand, 0, 1)[0]
    return C

def ContourEllipse(F,x,y,theta):
    ''' Calculate ellipses using Eq2 and 4 from https://arxiv.org/pdf/0906.4123.pdf'''
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
    ang = 0.5*np.arctan( 2*Cxy / (Cxx - Cyy) )
    ### Create ellipse equation line for plotting
    phi = np.linspace(0, 2*np.pi, 100) # angle variables
    Ellps = np.array([w*np.cos(phi) , h*np.sin(phi)]) # Ellipse equation
    # Construct 2D rotation matrix for ellipse:
    R_rot = np.array([[np.cos(ang) , -np.sin(ang)],[np.sin(ang) , np.cos(ang)]])
    Ellps_rot = np.zeros((2,Ellps.shape[1]))
    for i in range(Ellps.shape[1]):
        Ellps_rot[:,i] = np.dot(R_rot,Ellps[:,i])
    return theta[y]+Ellps_rot[1,:],theta[x]+Ellps_rot[0,:] # return reversed for matplotlib (j,i) panel convention

def CornerPlot(Fs,theta,theta_labels):
    if len(np.shape(Fs))==2: Fs = [Fs] # just one matrix provided.
    Npar = np.shape(Fs[0])[0]
    fig = plt.figure(figsize=(8,8))
    gs = GridSpec(Npar,Npar) # rows,columns
    for i in range(Npar):
        for j in range(Npar):
            if j>i: continue # only plot one corner of panels
            ax = fig.add_subplot(gs[i,j]) # First row, first column
            for F in Fs:
                C = np.linalg.inv(F)
                if i==(Npar-1): ax.set_xlabel(theta_labels[j])
                if j==0: ax.set_ylabel(theta_labels[i])
                if i==j: # Plot Gaussian distribution for marginalised parameter estimate
                    sigma = np.sqrt(C[i,i])
                    xi = np.linspace(theta[i]-5*sigma, theta[i]+5*sigma, 200)
                    gauss = stats.norm.pdf(xi, theta[i], sigma)
                    gauss /= np.max(gauss) # normalise so max =1
                    ax.plot(xi,gauss)
                    ax.set_ylim(bottom=0)
                    ax.set_yticks([])
                    if theta[i]==0: title = theta_labels[i]+r'$=0\pm%s$'%np.round(sigma,3)
                    else: title = r'$\sigma($'+theta_labels[i]+r'$)/$'+theta_labels[i]+r'$=%s$'%(np.round(100*sigma/theta[i],3))+'%'
                    ax.set_title(title)
                    continue
                ell_x,ell_y = ContourEllipse(F,i,j,theta)
                ax.plot(ell_x,ell_y)
