import numpy as np
import cosmo
import scipy
from scipy import integrate
from scipy.special import legendre as Leg
v_21cm = 1420.405751#MHz

def Red2Freq(z):
    # Convert redshift to frequency for HI emission (freq in MHz)
    return v_21cm / (1+z)

def Freq2Red(v):
    # Convert frequency to redshift for HI emission (freq in MHz)
    return (v_21cm/v) - 1

def get_param_vals(ids,z,cosmopars):
    ''' return model values for parameter strings '''
    Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL = cosmopars
    vals=[]
    Npar = len(ids)
    for i in range(Npar):
        if ids[i]==r'$\overline{T}_{\rm HI}$': vals.append( Tbar(z,Omega_HI) )
        if ids[i]==r'$b_{\rm HI}$': vals.append( b_HI )
        if ids[i]==r'$f$': vals.append( f )
        if ids[i]==r'$D_A$': vals.append( D_A )
        if ids[i]==r'$H$': vals.append( H )
        if ids[i]==r'$A$': vals.append( A )
        if ids[i]==r'$f_{\rm NL}$': vals.append( 0 )
    return np.array(vals)

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
    d_c = cosmo.D_com(z)
    delta_lz = cosmo.D_com(Freq2Red(nu1)) - d_c
    pix_area = (d_c * np.radians(np.sqrt(A_pix)) )**2 # [Mpc/h]^2 based on fixed pixel size in deg
    V_cell = pix_area * delta_lz
    P_N = V_cell * sigma_N**2
    return P_N

def B_beam(mu,k,R_beam):
    ''' Gaussian beam model '''
    if R_beam==0: return 1
    return np.exp( -(1-mu**2)*k**2*R_beam**2/2 )

def APpars(k_f,mu_f,D_A_f,H_f,z):
    a_perp, a_para = cosmo.D_A(z)/D_A_f , H_f / cosmo.H(z) # alpha AP parameters
    F_AP = a_para/a_perp
    k = k_f/a_perp * np.sqrt( 1 + mu_f**2*(F_AP**(-2)-1) )
    mu = mu_f/F_AP / np.sqrt( 1 + mu_f**2*(F_AP**(-2)-1) )
    return a_para,a_perp,k,mu

def P(k_f,mu_f,z,Pmod,cosmopars,surveypars,tracer='HI'):
    ''' 2D signal model for power spectrum '''
    ### _f subscripts mark the "measured" parameters based on fiducial cosmology assumed
    Omega_HI,b_HI,b_g,f,D_A_f,H_f,A,bphiHI,bphig,f_NL = cosmopars
    zmin,zmax,A_sky,V_bin,R_beam,t_tot,N_dish,P_N,nbar = surveypars
    a_para,a_perp,k,mu = APpars(k_f,mu_f,D_A_f,H_f,z)
    if tracer=='HI': return 1/a_para*1/a_perp**2 * Tbar(z,Omega_HI)**2 * (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) * B_beam(mu,k,R_beam)**2
    if tracer=='g': return 1/a_para*1/a_perp**2 * (b_g + f*mu**2 + bphig*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k)
    if tracer=='X': return 1/a_para*1/a_perp**2 * Tbar(z,Omega_HI) * (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1))*(b_g + f*mu**2 + bphig*f_NL*cosmo.M(k,z)**(-1)) * Pmod(k) * B_beam(mu,k,R_beam)

def P_obs(k_f,mu_f,z,Pmod,cosmopars,surveypars,tracer='HI'):
    ''' 2D observational power spectrum with noise components'''
    Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL = cosmopars
    zmin,zmax,A_sky,V_bin,R_beam,t_tot,N_dish,P_N,nbar = surveypars
    if tracer=='HI': return P(k_f,mu_f,z,Pmod,cosmopars,surveypars,tracer) + Tbar(z,Omega_HI)**2 * P_SN(z) * B_beam(mu_f,k_f,R_beam=0)**2 + P_N
    if tracer=='g': return P(k_f,mu_f,z,Pmod,cosmopars,surveypars,tracer) + 1/nbar
    if tracer=='X': return P(k_f,mu_f,z,Pmod,cosmopars,surveypars,tracer)

def P_ell(ell,k,z,Pmod,cosmopars,surveypars,tracer='HI'):
    ''' Integrate signal model over mu into multipole ell '''
    return (2*ell + 1) * integratePkmu(P,ell,k,z,Pmod,cosmopars,surveypars,tracer)

def P_ell_obs(ell,k,z,Pmod,cosmopars,surveypars,tracer='HI'):
    ''' Integrate observation model over mu into multipole ell '''
    return (2*ell + 1) * integratePkmu(P_obs,ell,k,z,Pmod,cosmopars,surveypars,tracer)

def integratePkmu(Pfunc,ell,k,z,Pmod,cosmopars,surveypars,tracer):
    '''integrate given Pfunc(k,mu) over mu with Legendre polynomial for given ell'''
    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    Pkmu = Pfunc(kgrid,mugrid,z,Pmod,cosmopars,surveypars,tracer) * Leg(ell)(mugrid)
    return scipy.integrate.simps(Pkmu, mu, axis=0) # integrate over mu axis (axis=0)

def P_err(k,mu,z,Pmod,cosmopars,surveypars,V_bin,tracer='HI'):
    dk = np.diff(k)
    if np.var(dk)/np.mean(dk)>1e-6: # use to detect non-linear k-bins
         print('\nError! - k-bins must be linearly spaced.'); exit()
    dk = np.mean(dk) # reduce array to a single number

    P_obs_ = P_obs(k,mu,z,Pmod,cosmopars,surveypars,tracer)

    if tracer=='HI' or tracer=='g': return np.sqrt( 2*(2*np.pi)**3/V_bin * 1/(4*np.pi*k**2*dk) ) * P_obs_
    if tracer=='X':
        P_HI = P_obs(k,mu,z,Pmod,cosmopars,surveypars,tracer='HI')
        P_g = P_obs(k,mu,z,Pmod,cosmopars,surveypars,tracer='g')
        #return 1/np.sqrt( (2*np.pi)**3/V_bin * 1/(4*np.pi*k**2*dk) ) * np.sqrt(P_obs_**2 + P_HI*P_g)
        return np.sqrt( (2*np.pi)**3/V_bin * 1/(4*np.pi*k**2*dk) ) * np.sqrt(P_obs_**2 + P_HI*P_g)

def P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,V_bin,tracer='HI'):
    ''' Integrate error model over mu into multipole ell '''
    return (2*ell + 1)**2 * integratePkmu_err(P_err,ell,k,z,Pmod,cosmopars,surveypars,V_bin,tracer)

def integratePkmu_err(Pfunc,ell,k,z,Pmod,cosmopars,surveypars,V_bin,tracer):
    '''integrate given Pfunc(k,mu) over mu with Legendre polynomial for given ell'''
    mu = np.linspace(0,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    Pkmu = Pfunc(kgrid,mugrid,z,Pmod,cosmopars,surveypars,V_bin,tracer) * Leg(ell)(mugrid)
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
