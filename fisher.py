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

epsilon = 1e-3 # differential step size **** CHECK FOR STABILITY ****
def dP_dtheta_stencil(k,mu,theta_id,tracer):
    '''Numerical derivation using five-point stencil method -
    see eq201 in Euclid paper: https://www.aanda.org/articles/aa/pdf/2020/10/aa38071-20.pdf'''
    # Define kicks to input parameter then calculate 5 point stencil numerical derivative
    kick_ind = np.zeros(len(cosmopars)) # binary mask to select input parameter to kick
    #if theta_id==r'$\overline{T}_{\rm HI}$':
    if theta_id==r'$b_1$': theta,kick,kick_ind[2] = b1,b1*epsilon,1
    if theta_id==r'$b_2$': theta,kick,kick_ind[3] = b2,b2*epsilon,1
    if theta_id==r'$b^\phi_1$': theta,kick,kick_ind[4] = bphi1,bphi1*epsilon,1
    if theta_id==r'$b^\phi_2$': theta,kick,kick_ind[5] = bphi2,bphi2*epsilon,1
    if theta_id==r'$f$': theta,kick,kick_ind[6] = f,f*epsilon,1
    if theta_id==r'$\alpha_\perp$': theta,kick,kick_ind[7] = a_perp,a_perp*epsilon,1
    if theta_id==r'$\alpha_\parallel$': theta,kick,kick_ind[8] = a_para,a_para*epsilon,1
    if theta_id==r'$A_{\rm BAO}$': return dPell_dtheta(ells,k,dlnP_dA,tau)
    if theta_id==r'$f_{\rm NL}$': theta,kick,kick_ind[10] = 1,epsilon,1 # set theta=1 and kick=epsilon for f_NL to avoid divide by zero and zero kick
    pp = cosmopars + kick_ind*kick
    pm = cosmopars - kick_ind*kick
    p2p = cosmopars + kick_ind*2*kick
    p2m = cosmopars - kick_ind*2*kick
    return 8*(model.P(k,mu,Pmod,pp,surveypars,tracer) - model.P(k,mu,Pmod,pm,surveypars,tracer) ) / (12*epsilon*theta) \
    - (model.P(k,mu,Pmod,p2p,surveypars,tracer) - model.P(k,mu,Pmod,p2m,surveypars,tracer) ) / (12*epsilon*theta)

def dlnP_dA(k,mu,tracer):
    return f_BAO/(1+A_BAO*f_BAO)

def Matrix_2D(theta_ids,k,Pmod_,cosmopars_,surveypars_,tracer='HI'):
    '''Compute full 2D anisotroic Fisher matrix for parameter set [theta]'''

    global z,Pmod,cosmopars,surveypars
    Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_
    global Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL

    #global V_bin
    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL = cosmopars
    z,V_bin1,V_bin2,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    V_bin = np.min([V_bin1,V_bin2])

    ### define mu and k bins and their spacing:
    mu = np.linspace(-1,1,300)
    mugrid,kgrid = np.meshgrid(mu,k)

    if tracer=='1' or tracer=='X' or tracer=='MT': Pa = model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,tracer='1')
    if tracer=='2' or tracer=='X' or tracer=='MT': Pb = model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,tracer='2')
    if tracer=='X' or tracer=='MT': Pab = model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,tracer='X')

    Npar = len(theta_ids)
    F = np.zeros((Npar,Npar))
    for i in range(Npar):
        for j in range(i,Npar):
            if tracer=='1' or tracer=='MT':
                dPa_dm = dP_dtheta_stencil(kgrid,mugrid,theta_ids[i],tracer='1')
                dPa_dn = dP_dtheta_stencil(kgrid,mugrid,theta_ids[j],tracer='1')
            if tracer=='2' or tracer=='MT':
                dPb_dm = dP_dtheta_stencil(kgrid,mugrid,theta_ids[i],tracer='2')
                dPb_dn = dP_dtheta_stencil(kgrid,mugrid,theta_ids[j],tracer='2')
            if tracer=='X' or tracer=='MT':
                dPab_dm = dP_dtheta_stencil(kgrid,mugrid,theta_ids[i],tracer='X')
                dPab_dn = dP_dtheta_stencil(kgrid,mugrid,theta_ids[j],tracer='X')
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

## Calculate the cross covariance between different power spectra.
## !!! Pa and Pb should include shot noise !!!
def cal_cov(Pa=0, Pab=0, Pb=0, tracer=None):
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

def dPell_dtheta(ells,k,theta_id,tau):
    '''**** CHANGE THIS DESCRIPTION ******* Generic derivitive multipole function, specify ln derivtive parameter model with derivfunc
        e.g. derivfunc = dlnP_dbHI for b_HI parameter'''
    ntau,nell,nk = len(tau),len(ells),len(k)
    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    res = np.zeros((ntau,nell*nk))
    subres = np.zeros((nell,nk))
    for t in range(ntau):
        for i,ell_i in enumerate(ells):
            if theta_id==r'$A_{\rm BAO}$':
                global f_BAO
                f_BAO = f_BAO_ell[t,i]
                integrand = (2*ell_i+1) * dlnP_dA(kgrid,mugrid,tau[t]) * model.P(kgrid,mugrid,Pmod,cosmopars,surveypars,tau[t]) * Leg(ell_i)(mugrid)
            else:
                integrand = (2*ell_i+1) * dP_dtheta_stencil(kgrid,mugrid,theta_id,tau[t]) * Leg(ell_i)(mugrid)
            ## Multiply by k so that 2 factors of deltaP/deltatheta in Fisher matrix picks up k^2 term.
            subres[i] = k * scipy.integrate.simps(integrand, mu, axis=0) # integrate over mu axis (axis=0)
        res[t] = np.ravel(subres)
    return np.ravel(res)


def Matrix_ell(theta_ids,k,Pmod_,cosmopars_,surveypars_,ells,tracer):
    '''Compute Fisher matrix for multipoles with parameter set [theta]'''

    global z,Pmod,cosmopars,surveypars
    Pmod=Pmod_; cosmopars=cosmopars_; surveypars=surveypars_
    global Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL

    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL = cosmopars
    z,V_bin1,V_bin2,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    V_bin = np.min([V_bin1,V_bin2])

    if tracer=='1': tau = ['1']
    if tracer=='2': tau = ['2']
    if tracer=='X': tau = ['X']
    if tracer=='MT': tau = ['1','2','X']

    ntau,nell,nk = len(tau),len(ells),len(k)

    #'''
    if r'$A_{\rm BAO}$' in theta_ids: # Only need to do if constraining A_BAO
        global f_BAO_ell
        f_BAO_ell = np.zeros((ntau,nell,len(k)))
        for t in range(ntau):
            for i,ell_i in enumerate(ells):
                f_BAO_ell[t,i] = model.Pk_noBAO(model.P_ell(ell_i,k,Pmod,cosmopars,surveypars,tau[t]),k)[1]
    #'''
    dk = np.diff(k)
    if np.var(dk)/np.mean(dk)>1e-6: # use to detect non-linear k-bins
         print('\nError! - k-bins must be linearly spaced.'); exit()
    dk = np.mean(dk) # reduce array to a single number
    Npar = len(theta_ids)

    Cinv = np.linalg.inv( Cov_ell(ells,k,Pmod,cosmopars,surveypars,tau) )
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
                #### THIS LOOP GIVES IDENTICAL RESULT AS ABOVE DOT PRODUCT OPERATION #####
                dPdth_i,dPdth_j = deriv_i(k),deriv_j(k)
                for s0 in range(nell*ntau):
                    for s1 in range(nell*ntau):
                        F[i,j] += np.sum( dk * dPdth_i[s0*nk:(s0+1)*nk] * np.diag(Cinv[s0*nk:(s0+1)*nk,s1*nk:(s1+1)*nk]) * dPdth_j[s1*nk:(s1+1)*nk] )
                '''
            if j<i: F[i,j] = F[j,i]
    F *= V_bin/(4*np.pi**2)
    return F

def Cov_ell(ells,k,Pmod,cosmopars,surveypars,tau):
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
                    if tau[t0]=='X' and tau[t1]=='X': integrand = (2*ell_i+1)*(2*ell_j+1)/2 * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * ( model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,'X')**2 + model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,'1')*model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,'2') )
                    else:
                        if tau[t0]==tau[t1]: integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,tau[t0])**2
                        else:
                            if tau[t0]=='X' or tau[t1]=='X': integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,tau[t0]) * model.P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,tau[t1])
                            else: integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P(kgrid,mugrid,Pmod,cosmopars,surveypars,'X')**2
                            # below ~equivalent:
                            #else: integrand = (2*ell_i+1)*(2*ell_j+1) * Leg(ell_i)(mugrid)*Leg(ell_j)(mugrid) * model.P(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tau[t0]) * model.P(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tau[t1])
                    # Calculating k's along diagonal of each multipole permutation in C
                    #  - first compute 1D diagonal array "C_diag" to place into broader C matrix:
                    C_diag = scipy.integrate.simps(integrand, mu, axis=0) # integrate over mu axis (axis=0)
                    C_sub[i*nk:i*nk+nk,j*nk:j*nk+nk] = np.identity(nk) * C_diag # bed 1D array along diagonal of multipole permutation in C
            C[ t0*nell*nk:(t0+1)*nell*nk, t1*nell*nk:(t1+1)*nell*nk ] = C_sub
    return C

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

#colors = ['tab:blue','tab:orange','tab:green','tab:red']
#colors = ['tab:blue','grey','tab:orange','tab:red']
def CornerPlot(Fs,theta,theta_labels,Flabels=None,ls=None,fontsize=14,legbox=[1.2,1.1]
                ,colors=['tab:blue','tab:orange','tab:green','tab:red']):
    if len(np.shape(Fs))==2: Fs = [Fs] # just one matrix provided.
    if ls is None: ls = ['-','-','-','-','-','-','-','-','-','-','-',]
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


                if i==(Npar-1): ax.set_xlabel(theta_labels[j],fontsize=fontsize+2)
                if j==0: ax.set_ylabel(theta_labels[i],fontsize=fontsize+2)
                if i==j: # Plot Gaussian distribution for marginalised parameter estimate
                    sigma = np.sqrt(C[i,i])
                    if np.isnan(sigma):
                        sigma = 0
                    xi = np.linspace(theta[i]-5*sigma, theta[i]+5*sigma, 200)
                    gauss = stats.norm.pdf(xi, theta[i], sigma)
                    gauss /= np.max(gauss) # normalise so max =1
                    if Flabels is None: ax.plot(xi,gauss)
                    else: ax.plot(xi,gauss,label=Flabels[Fi],color=colors[Fi],ls=ls[Fi])
                    ax.set_ylim(bottom=0,top=1.05)
                    ax.set_yticks([])
                    if theta[i]==0: title = theta_labels[i]+r'$=0\pm{:#.3g}$'.format(sigma)
                    else: title = r'$\sigma($'+theta_labels[i]+r'$)/$'+theta_labels[i]+r'$={:#.3g}$'.format(100*sigma/theta[i])+'%'
                    ax.set_title(title,fontsize=fontsize)
                    print( theta_labels[i]+r'$={:#.3g}\pm{:#.3g}$'.format( theta[i],sigma) )
                    if Fi==(len(Fs)-1):
                        if i==0 and Flabels is not None: ax.legend(bbox_to_anchor=[legbox[0],legbox[1]],fontsize=fontsize,frameon=False,loc='upper left',handlelength=1,handletextpad=0.4)
                        ax.axvline(theta[i],color='grey',lw=0.5,zorder=-1)
                    continue
                if F[i,j]!=0: # don't plot zero Fisher information infinite contours
                    ell_x,ell_y = ContourEllipse(F,i,j,theta)
                    ax.plot(ell_x,ell_y,color=colors[Fi],ls=ls[Fi])
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

############################################
####### LEGACY CODE NO LONGER USED: ########
############################################

'''
### Analytical derivitives:
def dlnP_dTbar(k,mu,tracer):
    if tracer=='HI': return 2 / model.Tbar(z,Omega_HI)
    if tracer=='g': return 0
    if tracer=='X': return 1 / model.Tbar(z,Omega_HI)
def dlnP_db1(k,mu,tracer):
    if tracer=='1': return 2 / (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))
    if tracer=='2': return 0
    if tracer=='X': return 1 / (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))
def dlnP_df(k,mu,tracer):
    if tracer=='1': return 2*mu**2 / (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))
    if tracer=='2': return 2*mu**2 / (b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1))
    if tracer=='X': return 2*mu**2 / (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1)) + 2*mu**2 / (b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1))
def dlnP_dA(k,mu,tracer):
    return f_BAO/(1+A*f_BAO)
'''
