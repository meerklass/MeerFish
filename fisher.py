import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy import linalg
import scipy.stats as stats
from scipy.special import legendre as Leg
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import model
import cosmo

def V_eff(k,mu,tracer):
    return V_bin * ( model.P(k,mu,z,Pmod,cosmopars,surveypars,tracer) / model.P_obs(k,mu,z,Pmod,cosmopars,surveypars,tracer) )**2

def dlnP_dTbar(k,mu,tracer):
    if tracer=='HI': return 2 / model.Tbar(z,Omega_HI)
    if tracer=='g': return 0
    if tracer=='X': return 1 / model.Tbar(z,Omega_HI)
def dlnP_dbHI(k,mu,tracer):
    if tracer=='HI': return 2 / (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1))
    if tracer=='g': return 0
    if tracer=='X': return 1 / (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1))
#'''
def dlnP_df(k,mu,tracer):
    if tracer=='HI': return 2*mu**2 / (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1))
    if tracer=='g': return 2*mu**2 / (b_g + f*mu**2 + bphig*f_NL*cosmo.M(k,z)**(-1))
    if tracer=='X': return 2*mu**2 / (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1)) + 2*mu**2 / (b_g + f*mu**2 + bphig*f_NL*cosmo.M(k,z)**(-1))
#'''
def dlnP_dA(k,mu,tracer):
    return f_BAO/(1+A*f_BAO)
'''
def dlnP_dfNL(k,mu,tracer):
    if tracer=='HI': return 2*bphiHI*cosmo.M(k,z)**(-1) / (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1))
    if tracer=='g': return 2*bphig*cosmo.M(k,z)**(-1) / (b_g + f*mu**2 + bphig*f_NL*cosmo.M(k,z)**(-1))
    if tracer=='X': return bphiHI*cosmo.M(k,z)**(-1) / (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1)) + bphig*cosmo.M(k,z)**(-1) / (b_g + f*mu**2 + bphig*f_NL*cosmo.M(k,z)**(-1))
'''
epsilon = 1e-5 # differential step size **** CHECK FOR STABILITY ****
#epsilon = 0.15
'''
def dlnP_df(k,mu,tracer):
    pp = [Omega_HI,b_HI,b_g,f*(1+epsilon),D_A,H,A,bphiHI,bphig,f_NL]
    pm = [Omega_HI,b_HI,b_g,f*(1-epsilon),D_A,H,A,bphiHI,bphig,f_NL]
    p2p = [Omega_HI,b_HI,b_g,f*(1+2*epsilon),D_A,H,A,bphiHI,bphig,f_NL]
    p2m = [Omega_HI,b_HI,b_g,f*(1-2*epsilon),D_A,H,A,bphiHI,bphig,f_NL]
    return 8*(np.log(model.P(k,mu,z,Pmod,pp,surveypars,tracer)) - np.log(model.P(k,mu,z,Pmod,pm,surveypars,tracer)) ) / (12*epsilon*f) \
    - (np.log(model.P(k,mu,z,Pmod,p2p,surveypars,tracer)) - np.log(model.P(k,mu,z,Pmod,p2m,surveypars,tracer)) ) / (12*epsilon*f)
'''
#'''
def dlnP_dfNL(k,mu,tracer):
    pp = [Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL+epsilon]
    pm = [Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL-epsilon]
    p2p = [Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL+2*epsilon]
    p2m = [Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL-2*epsilon]
    return 8*(np.log(model.P(k,mu,z,Pmod,pp,surveypars,tracer)) - np.log(model.P(k,mu,z,Pmod,pm,surveypars,tracer)) ) / (12*epsilon) \
    - (np.log(model.P(k,mu,z,Pmod,p2p,surveypars,tracer)) - np.log(model.P(k,mu,z,Pmod,p2m,surveypars,tracer)) ) / (12*epsilon)
#'''
def dlnP_dD_A(k,mu,tracer):
    pp = [Omega_HI,b_HI,b_g,f,D_A*(1+epsilon),H,A,bphiHI,bphig,f_NL]
    pm = [Omega_HI,b_HI,b_g,f,D_A*(1-epsilon),H,A,bphiHI,bphig,f_NL]
    p2p = [Omega_HI,b_HI,b_g,f,D_A*(1+2*epsilon),H,A,bphiHI,bphig,f_NL]
    p2m = [Omega_HI,b_HI,b_g,f,D_A*(1-2*epsilon),H,A,bphiHI,bphig,f_NL]
    return 8*(np.log(model.P(k,mu,z,Pmod,pp,surveypars,tracer)) - np.log(model.P(k,mu,z,Pmod,pm,surveypars,tracer)) ) / (12*epsilon*D_A) \
    - (np.log(model.P(k,mu,z,Pmod,p2p,surveypars,tracer)) - np.log(model.P(k,mu,z,Pmod,p2m,surveypars,tracer)) ) / (12*epsilon*D_A)
def dlnP_dH(k,mu,tracer):
    pp = [Omega_HI,b_HI,b_g,f,D_A,H*(1+epsilon),A,bphiHI,bphig,f_NL]
    pm = [Omega_HI,b_HI,b_g,f,D_A,H*(1-epsilon),A,bphiHI,bphig,f_NL]
    p2p = [Omega_HI,b_HI,b_g,f,D_A,H*(1+2*epsilon),A,bphiHI,bphig,f_NL]
    p2m = [Omega_HI,b_HI,b_g,f,D_A,H*(1-2*epsilon),A,bphiHI,bphig,f_NL]
    return 8*(np.log(model.P(k,mu,z,Pmod,pp,surveypars,tracer)) - np.log(model.P(k,mu,z,Pmod,pm,surveypars,tracer)) ) / (12*epsilon*H) \
    - (np.log(model.P(k,mu,z,Pmod,p2p,surveypars,tracer)) - np.log(model.P(k,mu,z,Pmod,p2m,surveypars,tracer)) ) / (12*epsilon*H)

def dPell_dtheta(ells,k,derivfunc,tau):
    '''Generic derivitive multipole function, specify ln derivtive parameter model with derivfunc
        e.g. derivfunc = dlnP_dbHI for b_HI parameter'''
    ntau,nell,nk = len(tau),len(ells),len(k)
    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    res = np.zeros((ntau,nell*nk))
    subres = np.zeros((nell,nk))
    for t in range(ntau):
        for i,ell_i in enumerate(ells):
            if derivfunc is dlnP_dA:
                global f_BAO
                f_BAO = f_BAO_ell[t,i]
            integrand = (2*ell_i+1) * derivfunc(kgrid,mugrid,tau[t]) * model.P(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tau[t]) * Leg(ell_i)(mugrid)
            ## Multiply by k so that 2 factors of deltaP/deltatheta in Fisher matrix picks up k^2 term.
            subres[i] = k * scipy.integrate.simps(integrand, mu, axis=0) # integrate over mu axis (axis=0)
        res[t] = np.ravel(subres)
    return np.ravel(res)

def Matrix_ell(theta_ids,k,Pmod_,z_,cosmopars_,surveypars_,V_bin_,ells=[0,2,4],tracer='HI'):
    '''Compute Fisher matrix for multipoles with parameter set [theta]'''

    global V_bin,z,Pmod,cosmopars,surveypars
    V_bin=V_bin_; z=z_; Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_

    global Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL
    Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL = cosmopars

    if tracer=='HI': tau = ['HI']
    if tracer=='g': tau = ['g']
    if tracer=='X': tau = ['X']
    if tracer=='MT': tau = ['HI','g','X']

    ntau,nell,nk = len(tau),len(ells),len(k)
    global f_BAO_ell
    f_BAO_ell = np.zeros((ntau,nell,len(k)))
    for t in range(ntau):
        for i,ell_i in enumerate(ells):
            f_BAO_ell[t,i] = model.Pk_noBAO(model.P_ell(ell_i,k,z,Pmod,cosmopars,surveypars,tau[t]),k)[1]

    dk = np.diff(k)
    if np.var(dk)/np.mean(dk)>1e-6: # use to detect non-linear k-bins
         print('\nError! - k-bins must be linearly spaced.'); exit()
    dk = np.mean(dk) # reduce array to a single number
    Npar = len(theta_ids)
    global deriv_i; global deriv_j

    Cinv = np.linalg.inv( Cov_ell(ells,k,z,Pmod,cosmopars,surveypars,tau) )
    '''
    C = Cov_ell(ells,k,z,Pmod,cosmopars,surveypars,tau)
    print(len(k))
    print(np.shape(C))
    plt.imshow(np.log10(C))
    nlines = ntau*nell
    for i in range(nlines):
        plt.axhline((i+1)*len(k),color='black',lw=1)
        plt.axvline((i+1)*len(k),color='black',lw=1)
    plt.colorbar()
    plt.xlim(0,nk*nell*ntau)
    plt.ylim(0,nk*nell*ntau)
    plt.show()
    exit()
    '''
    F = np.zeros((Npar,Npar)) # Full Fisher matrix summed over tracers
    for i in range(Npar):
        def deriv_i(k):
            if theta_ids[i]==r'$\overline{T}_{\rm HI}$': return dPell_dtheta(ells,k,dlnP_dTbar,tau)
            if theta_ids[i]==r'$b_{\rm HI}$': return dPell_dtheta(ells,k,dlnP_dbHI,tau)
            if theta_ids[i]==r'$f$': return dPell_dtheta(ells,k,dlnP_df,tau)
            if theta_ids[i]==r'$D_A$': return dPell_dtheta(ells,k,dlnP_dD_A,tau)
            if theta_ids[i]==r'$H$': return dPell_dtheta(ells,k,dlnP_dH,tau)
            if theta_ids[i]==r'$A$': return dPell_dtheta(ells,k,dlnP_dA,tau)
            if theta_ids[i]==r'$f_{\rm NL}$': return dPell_dtheta(ells,k,dlnP_dfNL,tau)
        for j in range(Npar):
            if j>=i: # avoid calculating symmetric off-diagonals twice
                def deriv_j(k):
                    if theta_ids[j]==r'$\overline{T}_{\rm HI}$': return dPell_dtheta(ells,k,dlnP_dTbar,tau)
                    if theta_ids[j]==r'$b_{\rm HI}$': return dPell_dtheta(ells,k,dlnP_dbHI,tau)
                    if theta_ids[j]==r'$f$': return dPell_dtheta(ells,k,dlnP_df,tau)
                    if theta_ids[j]==r'$D_A$': return dPell_dtheta(ells,k,dlnP_dD_A,tau)
                    if theta_ids[j]==r'$H$': return dPell_dtheta(ells,k,dlnP_dH,tau)
                    if theta_ids[j]==r'$A$': return dPell_dtheta(ells,k,dlnP_dA,tau)
                    if theta_ids[j]==r'$f_{\rm NL}$': return dPell_dtheta(ells,k,dlnP_dfNL,tau)
                # Sum over ell and integrate over k in one big matrix operation:
                F[i,j] += dk * np.dot( np.dot( deriv_i(k),Cinv ) , deriv_j(k) )
                '''
                #### THIS LOOP GIVES IDENTICAL RESULT AS ABOVE DOT PRODUCE OPERATION #####
                dPdth_i,dPdth_j = deriv_i(k),deriv_j(k)
                for s0 in range(nell*ntau):
                    for s1 in range(nell*ntau):
                        F[i,j] += np.sum( dk * dPdth_i[s0*nk:(s0+1)*nk] * np.diag(Cinv[s0*nk:(s0+1)*nk,s1*nk:(s1+1)*nk]) * dPdth_j[s1*nk:(s1+1)*nk] )
                '''
            if j<i: F[i,j] = F[j,i]
    F *= V_bin/(4*np.pi**2)
    return F

def Cov_ell(ells,k,z,Pmod,cosmopars,surveypars,tau):
    ''' (n_ell * nk) X (n_ell * nk) multi-tracer [t1xt2] covariance matrix for multipoles where each
    element integrates over mu '''
    nell,nk = len(ells),len(k)
    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    ntau = len(tau)
    C = np.zeros((ntau*nell*nk,ntau*nell*nk))
    for t0 in range(ntau):
        for t1 in range(ntau):
            C_sub = np.zeros((nell*nk,nell*nk))
            for i,ell_i in enumerate(ells):
                for j,ell_j in enumerate(ells):
                    # Get correct covariance matrix depending on tracer combination:
                    if tau[t0]=='X' and tau[t1]=='X': integrand = (2*ell_i+1)*(2*ell_j+1)/2 * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * ( model.P_obs(kgrid,mugrid,z,Pmod,cosmopars,surveypars,'X')**2 + model.P_obs(kgrid,mugrid,z,Pmod,cosmopars,surveypars,'HI')*model.P_obs(kgrid,mugrid,z,Pmod,cosmopars,surveypars,'g') )
                    else:
                        if tau[t0]==tau[t1]: integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P_obs(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tau[t0])**2
                        else:
                            if tau[t0]=='X' or tau[t1]=='X': integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P_obs(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tau[t0]) * model.P_obs(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tau[t1])
                            else: integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P(kgrid,mugrid,z,Pmod,cosmopars,surveypars,'X')**2
                            # below ~equivalent:
                            #else: integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tau[t0]) * model.P(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tau[t1])
                    # Calculating k's along diagonal of each multipole permutation in C
                    #  - first compute 1D diagonal array "C_diag" to place into broader C matrix:
                    C_diag = scipy.integrate.simps(integrand, mu, axis=0) # integrate over mu axis (axis=0)
                    C_sub[i*nk:i*nk+nk,j*nk:j*nk+nk] = np.identity(nk) * C_diag # bed 1D array along diagonal of multipole permutation in C
            C[ t0*nell*nk:(t0+1)*nell*nk, t1*nell*nk:(t1+1)*nell*nk ] = C_sub
    return C


def Matrix_2D(theta_ids,k,Pmod_,z_,cosmopars_,surveypars_,V_bin_,tracer='HI'):
    '''Compute full 2D anisotroic Fisher matrix for parameter set [theta]'''

    global V_bin,z,Pmod,cosmopars,surveypars
    V_bin=V_bin_; z=z_; Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_

    global Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL
    Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL = cosmopars

    ### define mu and k bins and their spacing:
    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    dk = np.diff(k)
    if np.var(dk)/np.mean(dk)>1e-6: # use to detect non-linear k-bins
         print('\nError! - k-bins must be linearly spaced.'); exit()
    dk = np.mean(dk) # reduce array to a single number
    dmu = np.diff(mu)[0]

    ########################################################
    #### CAN V_eff be moved outside loop ???? #######
    ########################################################

    Npar = len(theta_ids)
    F = np.zeros((Npar,Npar))
    global deriv_i; global deriv_j
    for i in range(Npar):
        def deriv_i(k_i,mu_i):
            if theta_ids[i]==r'$\overline{T}_{\rm HI}$': return dlnP_dTbar(k_i,mu_i)
            if theta_ids[i]==r'$b_{\rm HI}$': return dlnP_dbHI(k_i,mu_i)
            if theta_ids[i]==r'$f$': return dlnP_df(k_i,mu_i)
            if theta_ids[i]==r'$D_A$': return dlnP_dD_A(k_i,mu_i)
            if theta_ids[i]==r'$H$': return dlnP_dH(k_i,mu_i)
            if theta_ids[i]==r'$A$': return 1 # cannot spline fit for 2DPk
            if theta_ids[i]==r'$f_{\rm NL}$': return dlnP_dfNL(k_i,mu_i)
        for j in range(Npar):
            if j>=i: # avoid calculating symmetric off-diagonals twice
                def deriv_j(k_i,mu_i):
                    if theta_ids[j]==r'$\overline{T}_{\rm HI}$': return dlnP_dTbar(k_i,mu_i)
                    if theta_ids[j]==r'$b_{\rm HI}$': return dlnP_dbHI(k_i,mu_i)
                    if theta_ids[j]==r'$f$': return dlnP_df(k_i,mu_i)
                    if theta_ids[j]==r'$D_A$': return dlnP_dD_A(k_i,mu_i)
                    if theta_ids[j]==r'$H$': return dlnP_dH(k_i,mu_i)
                    if theta_ids[j]==r'$A$': return 1 # cannot spline fit for 2DPk
                    if theta_ids[j]==r'$f_{\rm NL}$': return dlnP_dfNL(k_i,mu_i)
                dFkmu = kgrid**2*deriv_i(kgrid,mugrid)*deriv_j(kgrid,mugrid)*V_eff(kgrid,mugrid,tracer)
                F[i,j] = 1/(4*np.pi**2) * dk*dmu * np.sum(dFkmu) # integrate (sum) over k and mu
            else: F[i,j] = F[j,i]
    return F

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
    if Cxx==Cyy: ang = 0.5*np.arctan(1e30) # avoid divide by zero for diagonal params p1=p2
    else: ang = 0.5*np.arctan( 2*Cxy / (Cxx - Cyy) )
    ### Create ellipse equation line for plotting
    phi = np.linspace(0, 2*np.pi, 100) # angle variables
    Ellps = np.array([w*np.cos(phi) , h*np.sin(phi)]) # Ellipse equation
    # Construct 2D rotation matrix for ellipse:
    R_rot = np.array([[np.cos(ang) , -np.sin(ang)],[np.sin(ang) , np.cos(ang)]])
    Ellps_rot = np.zeros((2,Ellps.shape[1]))
    for i in range(Ellps.shape[1]):
        Ellps_rot[:,i] = np.dot(R_rot,Ellps[:,i])
    return theta[y]+Ellps_rot[1,:],theta[x]+Ellps_rot[0,:] # return reversed for matplotlib (j,i) panel convention

colors = ['tab:blue','tab:orange','tab:green','tab:red']
colors = ['tab:blue','grey','tab:orange','tab:red']
def CornerPlot(Fs,theta,theta_labels,Flabels=None):
    if len(np.shape(Fs))==2: Fs = [Fs] # just one matrix provided.
    Npar = np.shape(Fs[0])[0]
    fig = plt.figure(figsize=(8,8))
    gs = GridSpec(Npar,Npar) # rows,columns
    xmins,xmaxs,ymins,ymaxs = np.zeros((Npar,Npar)),np.zeros((Npar,Npar)),np.zeros((Npar,Npar)),np.zeros((Npar,Npar))
    ell_xs,ell_ys = np.zeros((len(Fs),100)),np.zeros((len(Fs),100))
    for i in range(Npar):
        for j in range(Npar):
            if j>i: continue # only plot one corner of panels
            for Fi in range(len(Fs)):
                F = Fs[Fi]
                ell_xs[Fi],ell_ys[Fi] = ContourEllipse(F,i,j,theta)
            xmins[i,j] = np.min(ell_xs); ymins[i,j] = np.min(ell_ys)
            xmaxs[i,j] = np.max(ell_xs); ymaxs[i,j] = np.max(ell_ys)
            xbuff = np.abs(xmaxs[i,j] - xmins[i,j])*0.05
            ybuff = np.abs(ymaxs[i,j] - ymins[i,j])*0.05
            xmins[i,j] -= xbuff; ymins[i,j] -= ybuff
            xmaxs[i,j] += xbuff; ymaxs[i,j] += ybuff
    for i in range(Npar):
        for j in range(Npar):
            if j>i: continue # only plot one corner of panels
            ax = fig.add_subplot(gs[i,j]) # First row, first column
            ax.set_xlim(left=xmins[i,j],right=xmaxs[i,j])
            for Fi in range(len(Fs)):
                F = Fs[Fi]
                C = np.linalg.inv(F)
                if i==(Npar-1): ax.set_xlabel(theta_labels[j])
                if j==0: ax.set_ylabel(theta_labels[i])
                if i==j: # Plot Gaussian distribution for marginalised parameter estimate
                    sigma = np.sqrt(C[i,i])
                    xi = np.linspace(theta[i]-5*sigma, theta[i]+5*sigma, 200)
                    gauss = stats.norm.pdf(xi, theta[i], sigma)
                    gauss /= np.max(gauss) # normalise so max =1
                    if Flabels is None: ax.plot(xi,gauss)
                    else: ax.plot(xi,gauss,label=Flabels[Fi],color=colors[Fi])
                    ax.set_ylim(bottom=0)
                    ax.set_yticks([])
                    if theta[i]==0: title = theta_labels[i]+r'$=0\pm{:#.3g}$'.format(sigma)
                    else: title = r'$\sigma($'+theta_labels[i]+r'$)/$'+theta_labels[i]+r'$={:#.3g}$'.format(100*sigma/theta[i])+'%'
                    ax.set_title(title)
                    print( theta_labels[i]+r'$=0\pm{:#.3g}$'.format(sigma) )
                    if Fi==(len(Fs)-1):
                        if i==0 and Flabels is not None: ax.legend(bbox_to_anchor=[1.2,1.1],fontsize=12,frameon=False,loc='upper left')
                        ax.axvline(theta[i],color='grey',lw=0.5,zorder=-1)
                    continue
                ell_x,ell_y = ContourEllipse(F,i,j,theta)
                ax.plot(ell_x,ell_y,color=colors[Fi])
                ax.axvline(theta[j],color='grey',lw=0.5,zorder=-1)
                ax.axhline(theta[i],color='grey',lw=0.5,zorder=-1)
                ax.set_ylim(bottom=ymins[i,j],top=ymaxs[i,j])
            if j!=0: ax.set_yticks([])
            if i!=(Npar-1): ax.set_xticks([])
    plt.subplots_adjust(wspace=0, hspace=0, right=0.99,top=0.97)
