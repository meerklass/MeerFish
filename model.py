import numpy as np
import cosmo
import scipy
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import legendre as Leg
v_21cm = 1420.405751#MHz
c_km = 299792.458 #km/s

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
    return np.array(vals)

def get_param_selection(ids):
    theta_select=[]
    Npar = len(ids)
    for i in range(Npar):
        if ids[i]==r'$\overline{T}_{\rm HI}$': theta_select.append(0)
        if ids[i]==r'$\overline{T}_{\rm HI}$': theta_select.append(1)
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

def OmegaHI(z):
    ''' HI density parameter '''
     # Matches SKAO red book and Alkistis early papers also consistent with GBT
     #   Masui + Wolz measurements at z=0.8
    return 0.00048 + 0.00039*z - 0.000065*z**2

def Tbar(z,Omega_HI):
    ''' Mean HI temperature [Units of mK] '''
    Hz = cosmo.H(z) #km / Mpc s
    H0 = cosmo.H(0) #km / Mpc s
    h = H0/100
    return 180 * Omega_HI * h * (1+z)**2 / (Hz/H0)

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

def P_N(z,zmin,zmax,A_sky,t_tot,N_dish,T_sys=None,A_pix=0.3**2,deltanu=0.2,epsilon=1):
    ''' model for the noise power spectrum '''
    if t_tot is None: return 0 # no noise input
    nu = Red2Freq(z)
    nu1 = nu - deltanu # Use to calculate frequency channel width in Mpc/h for pixel volume
    if T_sys is None: # Calculate based on SKA red book eq1: https://arxiv.org/pdf/1811.02743.pdf
        Tspl = 3e3 #mK
        TCMB = 2.73e3 #mk
        Tgal = 25e3*(408/nu)**2.75
        #Trx = 15e3 + 30e3*(nu/1e3 - 0.75)**2 # From Red Book
        Trx = 7.5e3 + 10e3*(nu/1e3 - 0.75)**2 # Amended from above to better fit Wang+20 MK Pilot Survey
        T_sys = Trx + Tspl + TCMB + Tgal
    deltanu = deltanu * 1e6 # Convert MHz to Hz
    t_tot = t_tot * 60 * 60 # Convert observing hours to seconds
    N_p = A_sky / A_pix # Number of pointings[or pixels]
    t_p = N_dish * t_tot / N_p  # time per pointing
    sigma_N = T_sys / (epsilon * np.sqrt(2 * deltanu * t_p) ) # Santos+15 eq 5.1

    #### CURRENTLY FIXING DISTANCES FOR FIDUCIAL COSMOLOGY: ####
    d_c = cosmo.D_com(z)
    delta_lz = cosmo.D_com(Freq2Red(nu1)) - d_c
    ############################################################

    pix_area = (d_c * np.radians(np.sqrt(A_pix)) )**2 # [Mpc/h]^2 based on fixed pixel size in deg
    V_cell = pix_area * delta_lz
    P_N = V_cell * sigma_N**2
    return P_N

def B_beam(mu,k,z,theta_FWHM,cosmopars):
    ''' Gaussian beam model '''
    if theta_FWHM==0: return 1
    sig_beam = theta_FWHM/(2*np.sqrt(2*np.log(2))) # in degrees
    d_c = cosmo.D_com(z,cosmopars) # Comoving distance to frequency bin
    R_beam = d_c * np.radians(sig_beam) #Beam size [Mpc/h]
    return np.exp( -(1-mu**2)*k**2*R_beam**2/2 )

def B_zerr(mu,k,sigma_z,z,a_para=1):
    ### from eq 12: https://arxiv.org/pdf/2305.00404.pdf
    if sigma_z==0: return 1
    #Sigma_z = a_para * c_km * sigma_z / cosmo.H(z)
    Sigma_z = c_km * sigma_z / cosmo.H(z)
    return np.exp(-k**2*mu**2*Sigma_z**2/2)

def APpars(k_f,mu_f,a_perp,a_para):
    F_AP = a_para/a_perp
    k = k_f/a_perp * np.sqrt( 1 + mu_f**2*(F_AP**(-2)-1) )
    mu = mu_f/F_AP / np.sqrt( 1 + mu_f**2*(F_AP**(-2)-1) )
    return k,mu

def P(k_f,mu_f,Pmod,cosmopars,surveypars,tracer):
    ''' 2D signal model for power spectrum '''
    ### _f subscripts mark the "measured" parameters based on fiducial cosmology assumed
    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A,f_NL = cosmopars
    z,V_bin1,V_bin2,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    k,mu = APpars(k_f,mu_f,a_perp,a_para)
    alpha_v = 1/a_para*1/a_perp**2 # alpha factor to correct for the modification of the volume
    if tracer=='1': return alpha_v * Tbar1**2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) * B_beam(mu,k,z,theta_FWHM1,cosmopars)**2 * B_zerr(mu,k,sigma_z1,z,a_para)**2
    if tracer=='2': return alpha_v * Tbar2**2 * (b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) * B_beam(mu,k,z,theta_FWHM2,cosmopars)**2 * B_zerr(mu,k,sigma_z2,z,a_para)**2
    if tracer=='X': return alpha_v * Tbar1*Tbar2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))*(b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1)) * Pmod(k) * B_beam(mu,k,z,theta_FWHM1,cosmopars) * B_beam(mu,k,z,theta_FWHM2,cosmopars) * B_zerr(mu,k,sigma_z1,z,a_para) * B_zerr(mu,k,sigma_z2,z,a_para)

def P_obs(k_f,mu_f,Pmod,cosmopars,surveypars,tracer):
    ''' 2D observational power spectrum with noise components'''
    z,V_bin1,V_bin2,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A,f_NL = cosmopars

    #alpha_v = 1/a_para*1/a_perp**2 # alpha factor to correct for the modification of the volume
    alpha_v = 1 # Not include for noise term in Euclid prep paper

    if tracer=='1': return P(k_f,mu_f,Pmod,cosmopars,surveypars,tracer) +  alpha_v * P_N1
    if tracer=='2': return P(k_f,mu_f,Pmod,cosmopars,surveypars,tracer) +  alpha_v * P_N2
    if tracer=='X': return P(k_f,mu_f,Pmod,cosmopars,surveypars,tracer)

def P_ell(ell,k,Pmod,cosmopars,surveypars,tracer):
    ''' Integrate signal model over mu into multipole ell '''
    return (2*ell + 1) * integratePkmu(P,ell,k,Pmod,cosmopars,surveypars,tracer)

def P_ell_obs(ell,k,Pmod,cosmopars,surveypars,tracer):
    ''' Integrate observation model over mu into multipole ell '''
    return (2*ell + 1) * integratePkmu(P_obs,ell,k,Pmod,cosmopars,surveypars,tracer)

def integratePkmu(Pfunc,ell,k,Pmod,cosmopars,surveypars,tracer):
    '''integrate given Pfunc(k,mu) over mu with Legendre polynomial for given ell'''
    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    Pkmu = Pfunc(kgrid,mugrid,Pmod,cosmopars,surveypars,tracer) * Leg(ell)(mugrid)
    return scipy.integrate.simps(Pkmu, mu, axis=0) # integrate over mu axis (axis=0)

################################################################################################
###### CHECK P_err and P_ell_err functions - coded quickly ######################
################################################################################################
def Nmodes(k,mu,V_bin):
    '''
    NOT WORKING - GIVING SMALL NUMBERS
    '''
    dk = np.mean(np.diff(k[0,:]))
    dmu = np.mean(np.diff(mu[:,0]))
    print(dk)
    print(dmu)
    print(V_bin)
    return k**2*dk*dmu*V_bin / (8*np.pi**2)

def P_err(k,mu,z,Pmod,cosmopars,surveypars,tracer):
    P_obs_ = P_obs(k,mu,Pmod,cosmopars,surveypars,tracer)

    z,V_bin1,V_bin2,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2 = surveypars
    '''
    if tracer=='1' or tracer=='2': return P_obs_/np.sqrt(Nmodes(k,mu,V_bin))
    if tracer=='X':
        P_1 = P_obs(k,mu,Pmod,cosmopars,surveypars,tracer='1')
        P_2 = P_obs(k,mu,Pmod,cosmopars,surveypars,tracer='2')
        return np.sqrt( 1/(2*Nmodes(k,mu,V_bin)) * (P_obs_**2 + P_1*P_2) )
    '''
    dk = np.mean(np.diff(k[0,:]))
    if tracer=='1': return np.sqrt( 2*(2*np.pi)**3/V_bin1 * 1/(4*np.pi*k**2*dk) ) * P_obs_
    if tracer=='2': return np.sqrt( 2*(2*np.pi)**3/V_bin2 * 1/(4*np.pi*k**2*dk) ) * P_obs_
    if tracer=='X':
        V_binX = np.min([V_bin1,V_bin2])
        P_1 = P_obs(k,mu,Pmod,cosmopars,surveypars,tracer='1')
        P_2 = P_obs(k,mu,Pmod,cosmopars,surveypars,tracer='2')
        return np.sqrt( (2*np.pi)**3/V_binX * 1/(4*np.pi*k**2*dk) ) * np.sqrt(P_obs_**2 + P_1*P_2)

def P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,tracer):
    ''' Integrate error model over mu into multipole ell '''
    return (2*ell + 1)**2 * integratePkmu_err(P_err,ell,k,z,Pmod,cosmopars,surveypars,tracer)

################################################################################################
################################################################################################
def integratePkmu_err(Pfunc,ell,k,z,Pmod,cosmopars,surveypars,tracer):
    '''integrate given Pfunc(k,mu) over mu with Legendre polynomial for given ell'''
    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    Pkmu = Pfunc(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tracer) * Leg(ell)(mugrid)
    return scipy.integrate.simps(Pkmu, mu, axis=0) # integrate over mu axis (axis=0)

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
