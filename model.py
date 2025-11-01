import numpy as np
import cosmo
import scipy
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import legendre as Leg
v_21cm = 1420.405751#MHz
c_km = 299792.458 #km/s
import survey

def Red2Freq(z):
    # Convert redshift to frequency for HI emission (freq in MHz)
    return v_21cm / (1+z)

def Freq2Red(v):
    # Convert frequency to redshift for HI emission (freq in MHz)
    return (v_21cm/v) - 1

def get_param_vals(ids,z,cosmopars):
    ''' return model values for parameter strings '''
    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL = cosmopars
    vals=[]
    Npar = len(ids)
    for i in range(Npar):
        if ids[i]==r'$\overline{T}_{\rm HI}$': vals.append( Tbar1 )
        if ids[i]==r'$b_1$': vals.append( b1 )
        if ids[i]==r'$b_2$': vals.append( b2 )
        if ids[i]==r'$b^\phi_1$': vals.append( bphi1 )
        if ids[i]==r'$b^\phi_2$': vals.append( bphi2 )
        if ids[i]==r'$f$': vals.append( f )
        if ids[i]==r'$\alpha_\perp$': vals.append( a_perp )
        if ids[i]==r'$\alpha_\parallel$': vals.append( a_para )
        if ids[i]==r'$A_{\rm BAO}$': vals.append( A_BAO )
        if ids[i]==r'$f_{\rm NL}$': vals.append( f_NL )
        if ids[i]==r'$\delta_{\rm b}$': vals.append( 0 )
        if ids[i]==r'$\delta_{\rm sys}$': vals.append( 0 )
        if ids[i]==r'$\delta_{\rm z}$': vals.append( 0 )
    return np.array(vals)

'''
def get_param_selection(ids):
    theta_select=[]
    Npar = len(ids)
    for i in range(Npar):
        if ids[i]==r'$\overline{T}_{\rm HI}$': theta_select.append(0)
        if ids[i]==r'$\overline{T}_2$': theta_select.append(1)
        if ids[i]==r'$b_1$': theta_select.append( 2 )
        if ids[i]==r'$b_2$': theta_select.append( 3 )
        if ids[i]==r'$b^\phi_1$': theta_select.append( 4 )
        if ids[i]==r'$b^\phi_2$': theta_select.append( 5 )
        if ids[i]==r'$f$': theta_select.append( 6 )
        if ids[i]==r'$\alpha_\perp$': theta_select.append( 7 )
        if ids[i]==r'$\alpha_\parallel$': theta_select.append( 8 )
        if ids[i]==r'$A_{\rm BAO}$': theta_select.append( 9 )
        if ids[i]==r'$f_{\rm NL}$': theta_select.append( 10 )
    return np.array(theta_select)
'''

def b_HI(z):
    ''' HI linear bias '''
    '''Use 6 values for HI bias at redshifts 0 to 5 found in Table 5 of
    Villaescusa-Navarro et al.(2018) https://arxiv.org/pdf/1804.09180.pdf
    and get a polyfit function based on these values'''
    #### Code for finding polynomial coeficients: #####
    #z = np.array([0,1,2,3,4,5])
    #b_HI = np.array([0.84, 1.49, 2.03, 2.56, 2.82, 3.18])
    #coef = np.polyfit(z, b_HI,2)
    #A,B,C = coef[2],coef[1],coef[0]
    ###################################################
    A,B,C = 0.84178571,0.69289286,-0.04589286
    return A + B*z + C*z**2

def OmegaHI(z,SKAORedBook=False):
    ''' HI density parameter '''
    if SKAORedBook==True:
         # Matches SKAO red book and Alkistis early papers also consistent with GBT
         #   Masui + Wolz measurements at z=0.8
        return 0.00048 + 0.00039*z - 0.000065*z**2
    if SKAORedBook==False:
        # Shift first coeficient parameter (of Red Book model) to increase amplitude of
        #   whole spectrum so that it is more in line with recent measurements in WiggleZ-cross
        return 0.00067432 + 0.00039*z - 0.000065*z**2

def Tbar(z,Omega_HI):
    ''' Mean HI temperature [Units of mK] '''
    Hz = cosmo.H(z) #km / Mpc s
    H0 = cosmo.H(0) #km / Mpc s
    h = H0/100
    return 180 * Omega_HI * h * (1+z)**2 / (Hz/H0)

def get_kbins(z,zmin,zmax,A_sky,kmax=0.3):
    k_perp_min = np.pi / cosmo.D_ang(A_sky,z)
    k_para_min = np.pi / (cosmo.D_com(zmax)-cosmo.D_com(zmin))
    kmin = np.min([k_perp_min,k_para_min])
    kbins = np.arange(kmin,kmax,kmin)
    k = (kbins[1:] + kbins[:-1])/2 #centre of k bins
    return k,kbins,kmin,kmax

def P_SN(z):
    ''' HI shot noise '''
    '''Use 6 values for HI shot noise at redshifts 0 to 5 found in Table 5 of
    Villaescusa-Navarro et al.(2018) https://arxiv.org/pdf/1804.09180.pdf
    and get a polyfit function based on these values'''
    #### Code for finding polynomial coeficients: #####
    #z = np.array([0,1,2,3,4,5])
    #P_SN = np.array([104,124,65,39,14,7])
    #coef = np.polyfit(z, P_SN,4)
    #A,B,C,D,E = coef[4],coef[3],coef[2],coef[1],coef[0]
    ###################################################
    A,B,C,D,E = 104.76587301587332, 81.77513227513245, -87.78472222222258, 23.393518518518654, -1.9791666666666783
    return A + B*z + C*z**2 + D*z**3 + E*z**4

def P_N(z,A_sky,t_tot,N_dish,T_sys=None,A_pix=None,theta_FWHM=None,deltanu=0.2,epsilon=1,return_sigma=False):
    ''' model for radio thermal noise power spectrum '''
    ## This is (and should be) insensitive to choice of pixel area for the output
    ##   noise power, since 1/A_pix appears in radiometer but then cancelled by
    ##   A_pix in the volume factor of P_N. It changes output sigma_N though.
    ##   Same for delta_nu however n_chunk channels can be combined to increase
    ##   deltanu but then the t_obs also increases by n_chunk impacts the radiometer
    if t_tot is None: return 0 # no noise input
    deltanu_Hz = 1e6*deltanu # MHz -> Hz
    t_tot_sec = t_tot * 60 * 60 # Convert observing hours to seconds
    nu = Red2Freq(z)
    if A_pix is None:
        pix_size = theta_FWHM / 3 # [deg] based on MeerKAT pilot survey approach
        A_pix = pix_size**2 # Area covered in each pointing (related to beam size - equation formed by Steve)
    V_cell = survey.Vsur(Freq2Red(nu+deltanu/2),Freq2Red(nu-deltanu/2),A_pix)
    if T_sys is None: # Calculate based on SKA red book eq1: https://arxiv.org/pdf/1811.02743.pdf
        Tspl = 3e3 #mK
        TCMB = 2.73e3 #mk
        Tgal = 15e3*(408/nu)**2.75 # tuned to fit PySM with a cut at Dec<|10deg| galactic latitudes
        Trx = 7.5e3 + 10e3*(nu/1e3 - 0.75)**2 # Amended to fit Wang+20 MK Pilot Survey
        T_sys = Trx + Tspl + TCMB + Tgal
    N_p = A_sky / A_pix # Number of pointings[or pixels]
    t_p = N_dish * t_tot_sec / N_p  # time per pointing
    sigma_N = T_sys / (epsilon * np.sqrt(2 * deltanu_Hz * t_p) ) # Santos+15 eq 5.1
    P_N = V_cell * sigma_N**2
    if return_sigma==False: return P_N
    else: return sigma_N,P_N

def P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ''' 2D signal model for power spectrum '''
    ### dampsignal=True: will directly apply instrumental effects to signal (caution this can add non-cosmological information)
    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A,f_NL = cosmopars
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    dbeam,dsys,dphotoz = nuispars
    if dampsignal==False: # don't amply instrumental damping to signal power
        if tracer=='1': return Tbar1**2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) + dsys
        if tracer=='2': return Tbar2**2 * (b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k)
        if tracer=='X': return Tbar1*Tbar2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))*(b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1)) * Pmod(k)
    if dampsignal==True: # do amply instrumental damping to signal power
        if tracer=='1': return Tbar1**2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) * B_beam(mu,k,z,theta_FWHM1+dbeam)**2 * B_zerr(mu,k,sigma_z1,z)**2 + dsys
        if tracer=='2': return Tbar2**2 * (b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) * B_beam(mu,k,z,theta_FWHM2)**2 * B_zerr(mu,k,sigma_z2+dphotoz,z)**2
        if tracer=='X': return Tbar1*Tbar2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))*(b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1)) * Pmod(k) * B_beam(mu,k,z,theta_FWHM1+dbeam) * B_zerr(mu,k,sigma_z1,z) * B_beam(mu,k,z,theta_FWHM2) * B_zerr(mu,k,sigma_z2+dphotoz,z)

def P_ell(ell,k_1d,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ''' Integrate signal model over mu into multipole ell '''
    ### dampsignal=True: will directly apply instrumental effects to signal (caution this can add non-cosmological information)
    mu_1d = np.linspace(0,1,1000)
    k_m,mu_m = np.meshgrid(k_1d,mu_1d)
    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A,f_NL = cosmopars
    k_t,mu_t = APpars(k_m,mu_m,a_perp,a_para)
    alpha_v = 1/a_para*1/a_perp**2 # alpha factor to correct for the modification of the volume
    return alpha_v * (2*ell + 1) * scipy.integrate.simps( P(k_t,mu_t,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) * Leg(ell)(mu_m) , mu_1d, axis=0) # integrate over mu axis (axis=0)

def P_obs(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ### dampsignal=True: will not apply instrumental effects to noise since it remains in signal
    ''' 2D observational power spectrum with noise components'''
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    dbeam,dsys,dphotoz = nuispars
    if dampsignal==False: # damp noise terms to inflate errors
        if tracer=='1': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) + (P_N1 + dsys) / (B_beam(mu,k,z,theta_FWHM1+dbeam)**2 * B_zerr(mu,k,sigma_z1,z)**2)
        if tracer=='2': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) + (P_N2 + dsys) / (B_beam(mu,k,z,theta_FWHM2)**2 * B_zerr(mu,k,sigma_z2+dphotoz,z)**2)
        if tracer=='X': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal)
    if dampsignal==True: # don't apply damping to noise to inflate errors (signal damped instead)
        if tracer=='1': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) + (P_N1 + dsys)
        if tracer=='2': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) + (P_N2 + dsys)
        if tracer=='X': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal)

def P_ell_obs(ell,k,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ''' Integrate observation model over mu into multipole ell '''
    ### dampsignal=True: will not apply instrumental effects to noise since it remains in signal
    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    return (2*ell + 1) * scipy.integrate.simps( P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) * Leg(ell)(mugrid) , mu, axis=0) # integrate over mu axis (axis=0)

def sigma_error(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ### dampsignal=True: will not contain instrumental effects in noise (thus errors) since it remains in signal
    P_obs_ = P_obs(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal)
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    if tracer=='1': return P_obs_/np.sqrt(Nmodes(k,mu,V_bin1))
    if tracer=='2': return P_obs_/np.sqrt(Nmodes(k,mu,V_bin2))
    if tracer=='X':
        P_1 = P_obs(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer='1',dampsignal=dampsignal)
        P_2 = P_obs(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer='2',dampsignal=dampsignal)
        return np.sqrt( 1/(2*Nmodes(k,mu,V_binX)) * (P_obs_**2 + P_1*P_2) )

def sigma_ell_error(ell,k_1d,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ''' Integrate error model over mu into multipole ell '''
    ### dampsignal=True: will not contain instrumental effects in noise (thus errors) since it remains in signal
    mu_1d = np.linspace(0,1,1000)
    k,mu = np.meshgrid(k_1d,mu_1d)
    return np.sqrt( (2*ell + 1)**2 * scipy.integrate.simps( sigma_error(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal)**2 * Leg(ell)(mu)**2 , mu_1d, axis=0) ) # integrate over mu axis (axis=0)

def Nmodes(k,mu,V_bin):
    # leave out dmu term from standard definition otherwise its double counted
    #    in scipy.integrate.simps integral over mu when obtaining P_ell error
    dk = np.mean(np.diff(k[0,:]))
    return k**2*dk*V_bin / (8*np.pi**2)

def B_beam(mu,k,z,theta_FWHM):
    ''' Gaussian beam model '''
    if theta_FWHM==0: return 1
    sig_beam = theta_FWHM/(2*np.sqrt(2*np.log(2))) # in degrees
    d_c = cosmo.D_com(z) # fiducial comoving distance to surveys effective redshift
    R_beam = d_c * np.radians(sig_beam) #Beam size [Mpc/h]
    damp = np.exp( -(1-mu**2)*k**2*R_beam**2/2 )
    damp[damp<1e-30] = 1e-30 # set small value clip as was raising "RuntimeWarning: overflow encountered in add" error
    return damp

def B_zerr(mu,k,sigma_z,z):
    ### from eq 12: https://arxiv.org/pdf/2305.00404.pdf
    if sigma_z==0: return 1
    sigma_para = c_km * sigma_z / cosmo.H(z) # convert rms in redshift, to rms in comoving units
    damp = np.exp(-k**2*mu**2*sigma_para**2/2)
    damp[damp<1e-30] = 1e-30 # set small value clip as to avoid invalid value error
    return damp

def B_chan(mu,k,z,delta_nu=0):
    ### from eq 41: https://arxiv.org/pdf/1902.07439
    if delta_nu==0: return 1
    s_para = c_km/cosmo.H(z) * (1+z)**2 * delta_nu/v_21cm
    k_para = k*mu
    k_para[k_para==0] = 1e-30
    return np.sin(k_para*s_para/2) / (k_para*s_para/2)

def APpars(k_m,mu_m,a_perp,a_para):
    F_AP = a_para/a_perp
    k_t = k_m/a_perp * np.sqrt( 1 + mu_m**2*(F_AP**(-2)-1) )
    mu_t = mu_m/F_AP / np.sqrt( 1 + mu_m**2*(F_AP**(-2)-1) )
    return k_t,mu_t

def P_1D(k_m,mu_m,Pmod,cosmopars,surveypars,nuispars,tracer):
    '''
    UNDER DEVELOPMENT - NOT WORKING!!
    '''
    P3D = P(k_m,mu_m,Pmod,cosmopars,surveypars,nuispars,tracer)
    k_para = k_m*mu_m
    k_perp = k_m*np.sqrt(1-mu_m**2)
    nbins = 30
    # Choose bins in k_para
    kpara_bins = np.linspace(k_para.min(), k_para.max(), nbins+1)
    kpara_centers = 0.5*(kpara_bins[1:] + kpara_bins[:-1])
    P1D = np.zeros(nbins)
    for i in range(nbins):
        mask = (k_para >= kpara_bins[i]) & (k_para < kpara_bins[i+1])
        if np.any(mask):
            integrand = P3D[mask] * np.abs(k_perp[mask])
            P1D[i] = scipy.integrate.simps(integrand, k_perp[mask], axis=0) # integrate over mu axis (axis=0)
    return kpara_centers, P1D


def Pk_noBAO(Pk,k,kBAO=[0.03,0.3]):
    """ Code from Ze - following Phil Bull method (https://arxiv.org/pdf/1405.1452.pdf)
    Construct a smooth power spectrum with BAOs removed, and a corresponding
    BAO template function, by using a two-stage splining process."""
    if k[0]>kBAO[0]: print('ERROR! k-range outside BAO scales'); exit()
    if k[-1]<kBAO[-1]: print('ERROR! k-range outside BAO scales'); exit()
    # Get interpolating function for input P(k) in log-log space
    _interp_pk = scipy.interpolate.interp1d(np.log(k),np.log(Pk),kind='quadratic',bounds_error=False )
    interp_pk = lambda x: np.exp(_interp_pk(np.log(x)))
    # Spline all (log-log) points except those in user-defined "wiggle region",
    # and then get derivatives of result
    idxs = np.where(np.logical_or(k <= kBAO[0], k >= kBAO[1]))
    _pk_smooth = scipy.interpolate.UnivariateSpline(np.log(k[idxs]),np.log(Pk[idxs]),k=3,s=0)
    pk_smooth = lambda x: np.exp(_pk_smooth(np.log(x)))
    # Construct "wiggles" function using spline as a reference, then spline it
    # and find its 2nd derivative
    fwiggle = scipy.interpolate.UnivariateSpline(k,Pk/pk_smooth(k),k=3,s=0)
    derivs = np.array([fwiggle.derivatives(_k) for _k in k]).T
    d2 = scipy.interpolate.UnivariateSpline(k, derivs[2], k=3, s=1)
    # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
    # low-order spline through zeros to subtract smooth trend from wiggles fn.
    wzeros = d2.roots()
    wzeros = wzeros[np.where(np.logical_and(wzeros >= kBAO[0], wzeros <= kBAO[1]))]
    wzeros = np.concatenate(([kBAO[0],], wzeros)) # add first k from wiggle range for smooth trend stiching
    wzeros = np.concatenate((wzeros, [kBAO[1],])) # add last k from wiggle range for smooth trend stiching
    wtrend = scipy.interpolate.UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=0)
    # Construct smooth "no-bao" function by summing the original splined function and
    # the wiggles trend function
    idxs = np.where(np.logical_and(k > kBAO[0], k < kBAO[1]))
    Pk_smooth = pk_smooth(k)
    Pk_smooth[idxs] *= wtrend(k[idxs])
    f_BAO = (Pk - Pk_smooth)/Pk_smooth
    return Pk_smooth,f_BAO

def ChiSquare(x_obs,x_mod,x_err):
    #Reduced Chi-squared function
    return np.sum( ((x_obs-x_mod)/x_err)**2 )
