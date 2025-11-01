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

epsilon = 1e-2 # differential step size

def dPell_dtheta(ells,k,theta_id,tau):
    '''Obtain derivative vectors looping over tracer and multipoles'''
    ntau,nell,nk = len(tau),len(ells),len(k)
    res = np.zeros((ntau,nell*nk))
    subres = np.zeros((nell,nk))
    for t in range(ntau):
        for i,ell_i in enumerate(ells):
            ## Multiply by k so that 2 factors of deltaP/deltatheta in Fisher matrix picks up k^2 term.
            subres[i] = k * dPell_dtheta_stencil(k,ell_i,theta_id,tau[t])
        res[t] = np.ravel(subres)
    return np.ravel(res)

def dPell_dtheta_stencil(k,ell,theta_id,tracer):
    '''Numerical derivation using five-point stencil method -
    see eq201 in Euclid paper: https://www.aanda.org/articles/aa/pdf/2020/10/aa38071-20.pdf'''
    # Define kicks to input parameter then calculate 5 point stencil numerical derivative
    kick_ind = np.zeros(len(cosmopars)) # binary mask to select input parameter to kick
    if theta_id==r'$\overline{T}_{\rm HI}$': kick,kick_ind[0] = Tbar1*epsilon,1
    if theta_id==r'$b_1$': kick,kick_ind[2] = b1*epsilon,1
    if theta_id==r'$b_2$': kick,kick_ind[3] = b2*epsilon,1
    if theta_id==r'$b^\phi_1$': kick,kick_ind[4] = bphi1*epsilon,1
    if theta_id==r'$b^\phi_2$': kick,kick_ind[5] = bphi2*epsilon,1
    if theta_id==r'$f$': kick,kick_ind[6] = f*epsilon,1
    if theta_id==r'$\alpha_\perp$': kick,kick_ind[7] = a_perp*epsilon,1
    if theta_id==r'$\alpha_\parallel$': kick,kick_ind[8] = a_para*epsilon,1
    if theta_id==r'$A_{\rm BAO}$': return dPell_dtheta(ells,k,dlnP_dA,tau)
    if theta_id==r'$f_{\rm NL}$': kick,kick_ind[10] = epsilon,1 # set kick=epsilon for f_NL to avoid divide by zero and zero kick
    ## nuisance parameters:
    kick_ind_nuis = np.zeros(len(nuispars)) # binary mask to select input parameter to kick
    if theta_id==r'$\delta_{\rm b}$': kick,kick_ind[0] = epsilon*1e-2,1 # set kick=epsilon to avoid divide by zero and zero kick
    if theta_id==r'$\delta_{\rm sys}$': kick,kick_ind[1] = epsilon,1 # set kick=epsilon to avoid divide by zero and zero kick
    if theta_id==r'$\delta_{\rm z}$': kick,kick_ind[2] = epsilon*1e-2,1 # set kick=epsilon to avoid divide by zero and zero kick
    ## apply kick to chosen parameter and input into stencil:
    cpp = cosmopars + kick_ind*kick
    cpm = cosmopars - kick_ind*kick
    cp2p = cosmopars + kick_ind*2*kick
    cp2m = cosmopars - kick_ind*2*kick
    npp = nuispars + kick_ind_nuis*kick
    npm = nuispars - kick_ind_nuis*kick
    np2p = nuispars + kick_ind_nuis*2*kick
    np2m = nuispars - kick_ind_nuis*2*kick
    return (-1*model.P_ell(ell,k,Pmod,cp2p,surveypars,np2p,tracer,dampsignal) + 8*model.P_ell(ell,k,Pmod,cpp,surveypars,npp,tracer,dampsignal) \
     - 8*model.P_ell(ell,k,Pmod,cpm,surveypars,npm,tracer,dampsignal) + model.P_ell(ell,k,Pmod,cp2m,surveypars,np2m,tracer,dampsignal)) / (12*kick)

def dlnP_dA(k,mu,tracer):
    return f_BAO/(1+A_BAO*f_BAO)

def Matrix_ell(theta_ids,k,Pmod_,cosmopars_,surveypars_,nuispars_,ells,tracer,dampsignal_=True):
    '''Compute Fisher matrix for multipoles with parameter set [theta]'''

    global z,Pmod,cosmopars,surveypars,nuispars,dampsignal
    Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_; nuispars=nuispars_; dampsignal=dampsignal_
    global Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL

    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL = cosmopars
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    if tracer=='1': V_bin = V_bin1
    if tracer=='2': V_bin = V_bin2
    if tracer=='X' or tracer=='MT': V_bin = V_binX

    if tracer=='1': tau = ['1']
    if tracer=='2': tau = ['2']
    if tracer=='X': tau = ['X']
    if tracer=='MT': tau = ['1','2','X']

    ntau,nell,nk = len(tau),len(ells),len(k)

    if r'$A_{\rm BAO}$' in theta_ids: # Only need to do if constraining A_BAO
        global f_BAO_ell
        f_BAO_ell = np.zeros((ntau,nell,len(k)))
        for t in range(ntau):
            for i,ell_i in enumerate(ells):
                f_BAO_ell[t,i] = model.Pk_noBAO(model.P_ell(ell_i,k,Pmod,cosmopars,surveypars,nuispars,tau[t],dampsignal),k)[1]

    dk = np.diff(k)
    if np.var(dk)/np.mean(dk)>1e-6: # use to detect non-linear k-bins
         print('\nError! - k-bins must be linearly spaced.'); exit()
    dk = np.mean(dk) # reduce array to a single number
    Npar = len(theta_ids)

    Cinv = np.linalg.inv( Cov_ell(ells,k,Pmod,cosmopars,surveypars,nuispars,tau) )
    F = np.zeros((Npar,Npar)) # Full Fisher matrix summed over tracers
    for i in range(Npar):
        def deriv_i(k):
            return dPell_dtheta(ells,k,theta_ids[i],tau)
        for j in range(Npar):
            if j>=i: # avoid calculating symmetric off-diagonals twice
                def deriv_j(k):
                    return dPell_dtheta(ells,k,theta_ids[j],tau)
                # Sum over ell and integrate over k in one big matrix operation:
                F[i,j] += dk * np.dot( np.dot( deriv_i(k).T,Cinv ) , deriv_j(k) )
                '''
                #### SANITY CHECK: THIS LOOP GIVES IDENTICAL RESULT AS ABOVE DOT PRODUCT OPERATION #####
                dPdth_i,dPdth_j = deriv_i(k),deriv_j(k)
                for s0 in range(nell*ntau):
                    for s1 in range(nell*ntau):
                        F[i,j] += np.sum( dk * dPdth_i[s0*nk:(s0+1)*nk] * np.diag(Cinv[s0*nk:(s0+1)*nk,s1*nk:(s1+1)*nk]) * dPdth_j[s1*nk:(s1+1)*nk] )
                '''
            if j<i: F[i,j] = F[j,i]
    F *= V_bin/(4*np.pi**2)
    return F

def Cov_ell(ells,k,Pmod,cosmopars,surveypars,nuispars,tau):
    ''' (n_ell * nk) X (n_ell * nk) multi-tracer [t1xt2] reduced covariance matrix
    for multipoles where each element integrates over mu '''
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
                    if tau[t0]=='X' and tau[t1]=='X': integrand = (2*ell_i+1)*(2*ell_j+1)/2 * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * ( model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,'X',dampsignal)**2 + model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,'1',dampsignal)*model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,'2',dampsignal) )
                    else:
                        if tau[t0]==tau[t1]: integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,tau[t0],dampsignal)**2
                        else:
                            if tau[t0]=='X' or tau[t1]=='X': integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,tau[t0],dampsignal) * model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,tau[t1],dampsignal)
                            else: integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,'X',dampsignal)**2
                    # Calculating k's along diagonal of each multipole permutation in C
                    #  - first compute 1D diagonal array "C_diag" to place into broader C matrix:
                    C_diag = scipy.integrate.simps(integrand, mu, axis=0) # integrate over mu axis (axis=0)
                    C_sub[i*nk:i*nk+nk,j*nk:j*nk+nk] = np.identity(nk) * C_diag # bed 1D array along diagonal of multipole permutation in C
            C[ t0*nell*nk:(t0+1)*nell*nk, t1*nell*nk:(t1+1)*nell*nk ] = C_sub
    return C

### Full 2D P(k,mu) Fisher matrix - (without multipoles) ###
def Matrix_2D(theta_ids,k,Pmod_,cosmopars_,surveypars_,tracer='HI',dampsignal_=True):
    '''Compute full 2D anisotroic Fisher matrix for parameter set [theta]'''

    global z,Pmod,cosmopars,surveypars,nuispars,dampsignal
    Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_; nuispars=nuispars_; dampsignal=dampsignal_
    global Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL

    #global V_bin
    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL = cosmopars
    z,V_bin1,V_bin2, V_binX, theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    if tracer=='1': V_bin = V_bin1
    if tracer=='2': V_bin = V_bin2
    if tracer=='X' or tracer=='MT':V_bin = np.min([V_bin1,V_bin2])

    ### define mu and k bins and their spacing:
    mu = np.linspace(-1,1,300)
    mugrid,kgrid = np.meshgrid(mu,k)

    if tracer=='1' or tracer=='X' or tracer=='MT': Pa = model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,tracer='1',dampsignal=dampsignal)
    if tracer=='2' or tracer=='X' or tracer=='MT': Pb = model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,tracer='2',dampsignal=dampsignal)
    if tracer=='X' or tracer=='MT': Pab = model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,tracer='X',dampsignal=dampsignal)

    Npar = len(theta_ids)
    F = np.zeros((Npar,Npar))
    for i in range(Npar):
        for j in range(i,Npar):
            if tracer=='1' or tracer=='MT':
                dPa_dm = dP_dtheta_stencil(kgrid,mugrid,0,theta_ids[i],tracer='1')
                dPa_dn = dP_dtheta_stencil(kgrid,mugrid,0,theta_ids[j],tracer='1')
            if tracer=='2' or tracer=='MT':
                dPb_dm = dP_dtheta_stencil(kgrid,mugrid,0,theta_ids[i],tracer='2')
                dPb_dn = dP_dtheta_stencil(kgrid,mugrid,0,theta_ids[j],tracer='2')
            if tracer=='X' or tracer=='MT':
                dPab_dm = dP_dtheta_stencil(kgrid,mugrid,0,theta_ids[i],tracer='X')
                dPab_dn = dP_dtheta_stencil(kgrid,mugrid,0,theta_ids[j],tracer='X')

            F_k = np.zeros(len(k))
            for ki in range(len(k)):
                F_mu = np.zeros(len(mu))
                for mui in range(len(mu)):
                    if tracer=='1':
                        C = cal_cov(Pa[ki,mui],None,None,tracer)
                        dP_dm_array = np.array([dPa_dm[ki,mui]])
                        dP_dn_array = np.array([dPa_dn[ki,mui]])
                        inv_cov = 1/C
                    if tracer=='2':
                        C = cal_cov(None,None,Pb[ki,mui],tracer)
                        dP_dm_array = np.array([dPb_dm[ki,mui]])
                        dP_dn_array = np.array([dPb_dn[ki,mui]])
                        inv_cov = 1/C
                    if tracer=='X':
                        C = cal_cov(Pa[ki,mui],Pab[ki,mui],Pb[ki,mui],tracer)
                        dP_dm_array = np.array([dPab_dm[ki,mui]])
                        dP_dn_array = np.array([dPab_dn[ki,mui]])
                        inv_cov = 1/C
                    if tracer=='MT':
                        C = cal_cov(Pa[ki,mui], Pab[ki,mui], Pb[ki,mui], tracer)
                        dP_dm_array = np.array([dPa_dm[ki,mui], dPab_dm[ki,mui], dPb_dm[ki,mui]])
                        dP_dn_array = np.array([dPa_dn[ki,mui], dPab_dn[ki,mui], dPb_dn[ki,mui]])
                        inv_cov = linalg.inv(C)
                    F_mu[mui] = np.dot(np.dot(dP_dm_array, inv_cov), dP_dn_array)
                F_k[ki] = integrate.trapz(F_mu, mu) * V_bin * k[ki]**2/(8*np.pi**2)
            F[i,j] = integrate.trapz(F_k, k)
            F[j,i] = F[i,j] # fill in symmetric elements
    return F

def cal_cov(Pa=0, Pab=0, Pb=0, tracer=None):
    ## Pa and Pb should include noise
    if tracer=='1': cov = Pa*Pa
    if tracer=='2': cov = Pb*Pb
    if tracer=='X': cov = 0.5*(Pab*Pab + Pa*Pb)
    if tracer=='MT':
        cov = np.zeros((3,3))
        cov[0,0] = Pa * Pa
        cov[0,1] = Pa * Pab
        cov[0,2] = Pab* Pab
        cov[1,0] = Pa * Pab
        cov[1,1] = 0.5*(Pab*Pab + Pa*Pb)
        cov[1,2] = Pb * Pab
        cov[2,0] = Pab* Pab
        cov[2,1] = Pb * Pab
        cov[2,2] = Pb * Pb
    return cov

def apply_priors(F,priors,params):
    '''Apply priors to the finalised Fisher matrix'''
    # Give priors as % error which will be converted to parameter sigma and applied as prior
    # params confirm the parameters to be applied - needed due to nuisance parameters
    if np.shape(F)[0]!=len(priors):
        print('Error: Different number of priors provided for size of Fisher matrix')
        exit()
    for i in range(len(priors)):
        if priors[i] is None: continue
        prior_sigma = params[i] * (priors[i]/100)
        F[i,i] += 1 / prior_sigma**2
    return F

def ContourEllipse(F,x,y,theta):
    ''' Calculate ellipses using Eq2 and 4 from https://arxiv.org/pdf/0906.4123.pdf'''
    # F: input Fisher matrix
    # (x,y): chosen indices for parameter pair in Fisher matrix to compute elipse for
    #C = np.linalg.inv(F)
    C = FisherInverse(F)
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

def CornerPlot(Fs,theta,theta_labels,Flabels=None,ls=None,lw=None,fontsize=14,param_fontsize=None,legbox=[1.2,1.1]
                ,colors=['tab:blue','tab:orange','tab:green','tab:red'],doLegend=True,doTitle=True):
    if len(np.shape(Fs))==2: Fs = [Fs] # just one matrix provided.
    if ls is None: ls = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']
    if lw is None: lw = 2
    if param_fontsize is None: param_fontsize = fontsize + 2
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
                if F[i,j]==0: # Avoid zero Fisher information blowing up contour min/max
                    ell_xs[Fi],ell_ys[Fi] = theta[j],theta[i]
                else: ell_xs[Fi],ell_ys[Fi] = ContourEllipse(F,i,j,theta)
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

                #C = np.linalg.inv(F)
                C = FisherInverse(F)


                if i==(Npar-1): ax.set_xlabel(theta_labels[j],fontsize=param_fontsize)
                if j==0: ax.set_ylabel(theta_labels[i],fontsize=param_fontsize)
                if i==j: # Plot Gaussian distribution for marginalised parameter estimate
                    sigma = np.sqrt(C[i,i])
                    if np.isnan(sigma):
                        sigma = 0
                    xi = np.linspace(theta[i]-5*sigma, theta[i]+5*sigma, 200)
                    gauss = stats.norm.pdf(xi, theta[i], sigma)
                    gauss /= np.max(gauss) # normalise so max =1
                    if Flabels is None: ax.plot(xi,gauss,lw=lw)
                    else: ax.plot(xi,gauss,label=Flabels[Fi],color=colors[Fi],ls=ls[Fi],lw=lw)
                    ax.set_ylim(bottom=0,top=1.05)
                    ax.set_yticks([])
                    if theta[i]==0: title = theta_labels[i]+r'$=0\pm{:#.3g}$'.format(sigma)
                    else: title = r'$\sigma($'+theta_labels[i]+r'$)/$'+theta_labels[i]+r'$={:#.3g}$'.format(100*sigma/theta[i])+'%'
                    if doTitle==True: ax.set_title(title,fontsize=fontsize)
                    print( theta_labels[i]+r'$={:#.3g}\pm{:#.3g}$'.format( theta[i],sigma) )
                    if Fi==(len(Fs)-1):
                        if i==0 and Flabels is not None:
                            if doLegend==True: ax.legend(bbox_to_anchor=[legbox[0],legbox[1]],fontsize=fontsize,frameon=False,loc='upper left',handlelength=1,handletextpad=0.4)
                        ax.axvline(theta[i],color='grey',lw=0.5,zorder=-1)
                    continue
                if F[i,j]!=0: # don't plot zero Fisher information infinite contours
                    ell_x,ell_y = ContourEllipse(F,i,j,theta)
                    ax.plot(ell_x,ell_y,color=colors[Fi],ls=ls[Fi],lw=lw)
                ax.axvline(theta[j],color='grey',lw=0.5,zorder=-1)
                ax.axhline(theta[i],color='grey',lw=0.5,zorder=-1)

                ax.set_ylim(bottom=ymins[i,j],top=ymaxs[i,j])
            if j!=0: ax.set_yticks([])
            if i!=(Npar-1): ax.set_xticks([])
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    plt.subplots_adjust(wspace=0, hspace=0, right=0.99,top=0.97)

def FisherInverse(F):
    '''Function for calculating inverse Fisher matrix ignoring any zeros within
    it. First remove any zeros from Fisher matrix to avoid zeros from ruining
    inverse and rest of code, then reconstruct to original shape'''
    nonzerocount = np.count_nonzero(np.diag(F))
    Fnz = np.reshape( F[F!=0], (nonzerocount,nonzerocount) )
    Cnz = np.linalg.inv(Fnz) # no zero covariance
    C = np.zeros(np.shape(F))
    C[F!=0] = np.ravel(Cnz)
    return C

# ---------- 5-point stencil stability tooling ----------
from contextlib import contextmanager

@contextmanager
def _use_epsilon(temp_eps):
    """Temporarily set the global epsilon used by dP_dtheta_stencil."""
    global epsilon
    _old = epsilon
    epsilon = float(temp_eps)
    try:
        yield
    finally:
        epsilon = _old

def _deriv_vector(theta_id, ells, k, mu, tracer):
    """
    Build the same derivative vector your Fisher uses:
    concatenate dP_ell/dtheta over all ells and k (after integrating over mu).
    """
    out = []
    for ell in ells:
        # this matches how you build Fisher: integrate over mu then keep k-dependence
        integrand = dP_dtheta_stencil(k, mu, ell, theta_id, tracer)  # shape (len(mu), len(k))
        dPell = scipy.integrate.simps(integrand, mu, axis=0)         # -> (len(k),)
        out.append(np.ravel(dPell))
    return np.concatenate(out)  # shape (n_ell * n_k,)

def check_stencil_convergence(theta_id, ells, k, tracer, Pmod_, cosmopars_, surveypars_,
                              eps_grid=(2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3),
                              #eps_grid=(1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3),
                              tol_rel=1e-2):
    """
    Scan eps_grid, compute derivative vectors, and report:
      - aim for tol_rel (relative tolerance) default 1e-2 i.e. 1%.
           [means numerical derivitive is statble to within 1% when step size is halved]
           So the “recommended ε” is the largest step for which halving ε changes the derivative vector by <1%
      - ||Δ||/||D|| when halving ε,
    Returns a dict with results and a suggested epsilon.
    """
    global z,Pmod,cosmopars,surveypars
    Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_
    global Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL
    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL = cosmopars
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    if tracer=='1': V_bin = V_bin1
    if tracer=='2': V_bin = V_bin2
    if tracer=='X' or tracer=='MT': V_bin = V_binX

    if tracer=='1': tau = ['1']
    if tracer=='2': tau = ['2']
    if tracer=='X': tau = ['X']
    if tracer=='MT': tau = ['1','2','X']

    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    ntau,nell,nk = len(tau),len(ells),len(k)


    results = {}
    for eps in eps_grid:
        with _use_epsilon(eps):
            D = _deriv_vector(theta_id, ells, kgrid, mugrid, tracer)
        results[eps] = D

    # Compare successive pairs (eps -> eps/2)
    rows = []
    keys = list(results.keys())
    for i in range(len(keys)-1):
        e_big, e_small = keys[i], keys[i+1]
        D_big, D_small = results[e_big], results[e_small]
        # global relative change
        rel = np.linalg.norm(D_small - D_big) / max(1.0, np.linalg.norm(D_small))
        # elementwise robust statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_elem = np.abs((D_small - D_big) / np.where(np.abs(D_small)>0, np.abs(D_small), 1.0))
        med_rel = np.median(rel_elem)
        max_rel = np.max(rel_elem)
        rows.append((e_big, e_small, rel, med_rel, max_rel))

    '''
    # Check 1/16 behaviour (truncation error ∝ ε^4)
    R = []
    for i in range(len(keys)-2):
        e1, e2, e3 = keys[i], keys[i+1], keys[i+2]
        d12 = np.linalg.norm(results[e2] - results[e1])
        d23 = np.linalg.norm(results[e3] - results[e2])
        ratio = d23 / d12 if d12 != 0 else np.nan
        R.append((e1, e2, e3, ratio))

    # Pick recommended epsilon: the largest ε where rel < tol_rel
    recommended = None
    for (e_big, e_small, rel, med_rel, max_rel) in rows:
        if rel < tol_rel and med_rel < 2*tol_rel:
            recommended = e_small  # we accept the smaller one in the pair
            break
    if recommended is None:
        # fall back to the smallest scanned ε
        recommended = keys[-1]
    '''

    # Print a concise report
    print(f"\nStencil stability for {theta_id}: ells={ells}, n_k={len(k)}, n_mu={len(mu)}")
    print("Δ when halving ε  ->  ||Δ||/||D||   median(|Δ/D|)   max(|Δ/D|)")
    #'''
    for (e_big, e_small, rel, med_rel, max_rel) in rows:
        print(f"  {e_big:8.2e} → {e_small:8.2e} :  {rel:9.3e}   {med_rel:12.3e}   {max_rel:11.3e}")
    '''
    print(f"\nSuggested ε for {theta_id}: {recommended:.3e} (tol_rel={tol_rel})")
    return {
        "eps_grid": keys,
        "D": results,
        "pairs": rows,
        "ratios": R,
        "recommended": recommended,
    }
    '''
    return

def derivative_richardson(theta_id, ells, k, tracer,
                          mu=np.linspace(0.0, 1.0, 201),
                          eps=1e-2):
    """
    5-point stencil is O(ε^4). Richardson-extrapolate with ε and ε/2 to get O(ε^6):
      D_extrap = (16 D(ε/2) - D(ε)) / 15
    Also return an internal error estimate err ≈ |D_extrap - D(ε/2)|.
    """
    with _use_epsilon(eps):
        D1 = _deriv_vector(theta_id, ells, k, mu, tracer)
    with _use_epsilon(eps/2):
        D2 = _deriv_vector(theta_id, ells, k, mu, tracer)
    D_extrap = (16.0*D2 - D1) / 15.0
    err = np.abs(D_extrap - D2)
    return D_extrap, err
# -------------------------------------------------------
