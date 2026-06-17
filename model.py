import numpy as np
import cosmo
import scipy
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import legendre as Leg
from scipy.interpolate import splrep, splev
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
    amp1,amp2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL,beta1,beta2 = cosmopars
    vals=[]
    Npar = len(ids)
    for i in range(Npar):
        if ids[i]==r'$\mathcal{A}_1$': vals.append( amp1 )
        if ids[i]==r'$\mathcal{A}_2$': vals.append( amp2 )
        if ids[i]==r'$b_1$': vals.append( b1 )
        if ids[i]==r'$b_2$': vals.append( b2 )
        if ids[i]==r'$b^\phi_1$': vals.append( bphi1 )
        if ids[i]==r'$b^\phi_2$': vals.append( bphi2 )
        if ids[i]==r'$f$': vals.append( f )
        if ids[i]==r'$\alpha_\perp$': vals.append( a_perp )
        if ids[i]==r'$\alpha_\parallel$': vals.append( a_para )
        if ids[i]==r'$A_{\rm BAO}$': vals.append( A_BAO )
        if ids[i]==r'$f_{\rm NL}$': vals.append( f_NL )
        if ids[i]==r'$\beta_1$': vals.append( beta1 )
        if ids[i]==r'$\beta_2$': vals.append( beta2 )
        if ids[i]==r'$\delta_{\rm b}$': vals.append( 0 )
        if ids[i]==r'$\delta_{\rm sys}$': vals.append( 0 )
        if ids[i]==r'$\delta_{\rm z}$': vals.append( 0 )
    return np.array(vals)

def b_HI(z,ZesVersion=False):
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
    if ZesVersion==False: return A + B*z + C*z**2
    if ZesVersion==True: return 0.67 + 0.18*z + 0.05*z**2

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

def Tbar(z,Omega_HI,ZesVersion=False):
    ''' Mean HI temperature [Units of mK] '''
    if ZesVersion==False:
        Hz = cosmo.H(z) #km / Mpc s
        H0 = cosmo.H(0) #km / Mpc s
        h = H0/100
        return 180 * Omega_HI * h * (1+z)**2 / (Hz/H0)
    if ZesVersion==True:
        return 0.0559 + 0.2324*z - 0.024*z**2

def Tsys(z):
    """Ze's function: returns interpolated system temperature in mK"""
    #in MHz, K
    HI = 1420 #MHz
    nu,Tsys_eta,a_eff = np.loadtxt('/Users/sadmin/Documents/MeerFish/Tsys/UHF_Tsys.txt',unpack=True) # set to location of MeerFish
    ratio = 0.72
    T_sys_meerkat_rep = splrep(nu,Tsys_eta)
    T_sys_meerkat = splev( HI / (1+z), T_sys_meerkat_rep )
    return T_sys_meerkat*ratio * 1e3

def get_kbins(z,zmin,zmax,A_sky,kmax=0.3,Taruya=False):
    if Taruya==False:
        k_perp_min = np.pi / cosmo.D_ang(A_sky,z)
        k_para_min = np.pi / (cosmo.D_com(zmax)-cosmo.D_com(zmin))
        kmin = np.min([k_perp_min,k_para_min])
        kbins = np.arange(kmin,kmax,kmin)
    if Taruya==True:
        #kbins defined by kmin from Taruya+10 (https://arxiv.org/pdf/1006.0699) below eq29
        kmin = 2*np.pi / survey.Vsur(zmin,zmax,A_sky)**(1/3)
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

def apply_kcuts(k,mu,kcuts=None):
    '''Trim k and mu arrays according to kperp,kpara cuts'''
    #kcuts: [k_perp_min,k_para_min,k_perp_max,k_para_max]
    if kcuts is None: return 1 # no cuts
    k_para,k_perp = k*mu,k*np.sqrt(1-mu**2)
    mask = np.ones_like(k, dtype=bool) # True when within kcuts boundaries
    if kcuts[0] is not None:
        mask &= (k_perp > kcuts[0])
    if kcuts[1] is not None:
        mask &= (k_para > kcuts[1])
    if kcuts[2] is not None:
        mask &= (k_perp < kcuts[2])
    if kcuts[3] is not None:
        mask &= (k_para < kcuts[3])
    return mask

def P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ''' 2D signal model for power spectrum '''
    ### dampsignal=True: will directly apply instrumental effects to signal (caution this can add non-cosmological information)
    amp1,amp2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A,f_NL,beta1,beta2 = cosmopars
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2,k_fg,dpix,dnu,kcuts = surveypars
    dbeam,dsys,dphotoz = nuispars
    if dampsignal==False: # don't apply instrumental damping to signal power
        if beta1!=1 or beta2!=-1: # reparameterise with beta=f/b parameters
            # can't do f_NL in this case so these params ignored
            if tracer=='1': return amp1**2 * (1 + beta1*mu**2)**2 * Pmod(k) + dsys
            if tracer=='2': return amp2**2 * (1 + beta2*mu**2)**2 * Pmod(k)
            if tracer=='X': return amp1*amp2 * (1 + beta1*mu**2)*(1 + beta2*mu**2) * Pmod(k)
        else: # standard model with b and f separated
            if tracer=='1': return amp1**2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) + dsys
            if tracer=='2': return amp2**2 * (b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k)
            if tracer=='X': return amp1*amp2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))*(b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1)) * Pmod(k)
    if dampsignal==True: # do apply instrumental damping to signal power
        if beta1!=-1 or beta2!=-1: # reparameterise with beta=f/b parameters
            # can't do f_NL in this case so these params ignored
            if tracer=='1': return amp1**2 * (1 + beta1*mu**2)**2 * Pmod(k) \
                * B_beam(mu,k,z,theta_FWHM1+dbeam)**2 * B_zerr(mu,k,sigma_z1,z)**2 * B_fg(mu,k,k_fg) * B_grid(mu,k,z,cosmopars,dpix,dnu)**2 + dsys
            if tracer=='2': return amp2**2 * (1 + beta2*mu**2)**2 * Pmod(k) \
                * B_beam(mu,k,z,theta_FWHM2)**2 * B_zerr(mu,k,sigma_z2+dphotoz,z)**2 * B_grid(mu,k,z,cosmopars,dpix,dnu,novox=True)**2
            if tracer=='X': return amp1*amp2 * (1 + beta1*mu**2)*(1 + beta2*mu**2) * Pmod(k) \
                * B_beam(mu,k,z,theta_FWHM1+dbeam) * B_zerr(mu,k,sigma_z1,z) * B_beam(mu,k,z,theta_FWHM2) * B_zerr(mu,k,sigma_z2+dphotoz,z) * B_fg(mu,k,k_fg) * B_grid(mu,k,z,cosmopars,dpix,dnu) * B_grid(mu,k,z,cosmopars,dpix,dnu,novox=True)
        else: # standard model with b and f separated
            if tracer=='1': return amp1**2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) \
                * B_beam(mu,k,z,theta_FWHM1+dbeam)**2 * B_zerr(mu,k,sigma_z1,z)**2 * B_fg(mu,k,k_fg) * B_grid(mu,k,z,cosmopars,dpix,dnu)**2 + dsys
            if tracer=='2': return amp2**2 * (b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) \
                * B_beam(mu,k,z,theta_FWHM2)**2 * B_zerr(mu,k,sigma_z2+dphotoz,z)**2 * B_grid(mu,k,z,cosmopars,dpix,dnu,novox=True)**2
            if tracer=='X': return amp1*amp2 * (b1 + f*mu**2 + bphi1*f_NL*cosmo.M(k,z)**(-1))*(b2 + f*mu**2 + bphi2*f_NL*cosmo.M(k,z)**(-1)) * Pmod(k) \
                * B_beam(mu,k,z,theta_FWHM1+dbeam) * B_zerr(mu,k,sigma_z1,z) * B_beam(mu,k,z,theta_FWHM2) * B_zerr(mu,k,sigma_z2+dphotoz,z) * B_fg(mu,k,k_fg) * B_grid(mu,k,z,cosmopars,dpix,dnu) * B_grid(mu,k,z,cosmopars,dpix,dnu,novox=True)

def P_ell(ell,k_1d,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ''' Integrate signal model over mu into multipole ell '''
    ### dampsignal=True: will directly apply instrumental effects to signal (caution this can add non-cosmological information)
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2,k_fg,dpix,dnu,kcuts = surveypars
    mu_1d = np.linspace(-1,1,1000)
    k_m,mu_m = np.meshgrid(k_1d,mu_1d)
    amp1,amp2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A,f_NL,beta1,beta2 = cosmopars
    k_t,mu_t = APpars(k_m,mu_m,a_perp,a_para)
    alpha_v = 1/a_para*1/a_perp**2 # alpha factor to correct for the modification of the volume
    integrand = P(k_t,mu_t,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) * Leg(ell)(mu_m) * apply_kcuts(k_m,mu_m,kcuts)
    result = alpha_v * (2*ell + 1)/2 * scipy.integrate.simpson( integrand , x=mu_1d, axis=0) # integrate over mu axis (axis=0)
    return result[result!=0] # ensure no zero contributions after kcuts

def P_obs(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ### dampsignal=True: will not apply instrumental effects to noise since it remains in signal
    ''' 2D observational power spectrum with noise components'''    
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2,k_fg,dpix,dnu,kcuts = surveypars
    dbeam,dsys,dphotoz = nuispars
    if dampsignal==False: # damp noise terms to inflate errors
        if tracer=='1': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) + (P_N1 + dsys) \
            / (B_beam(mu,k,z,theta_FWHM1+dbeam)**2 * B_zerr(mu,k,sigma_z1,z)**2 * B_fg(mu,k,k_fg)) * B_grid(mu,k,z,cosmopars,dpix,dnu)**2
        if tracer=='2': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) + (P_N2 + dsys) \
            / (B_beam(mu,k,z,theta_FWHM2)**2 * B_zerr(mu,k,sigma_z2+dphotoz,z)**2) * B_grid(mu,k,z,cosmopars,dpix,dnu,novox=True)**2
        if tracer=='X': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal)
    if dampsignal==True: # don't apply damping to noise to inflate errors (signal damped instead)
        if tracer=='1': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) + (P_N1 + dsys) * B_grid(mu,k,z,cosmopars,dpix,dnu,novox=True)**2
        if tracer=='2': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) + (P_N2 + dsys)
        if tracer=='X': return P(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal)

def P_ell_obs(ell,k,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ''' Integrate observation model over mu into multipole ell '''
    ### dampsignal=True: will not apply instrumental effects to noise since it remains in signal
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2,k_fg,dpix,dnu,kcuts = surveypars
    mu = np.linspace(-1,1,1000)
    kgrid,mugrid = np.meshgrid(k,mu)
    integrand = P_obs(kgrid,mugrid,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal) * Leg(ell)(mugrid) * apply_kcuts(kgrid,mugrid,kcuts)
    result = (2*ell + 1)/2 * scipy.integrate.simpson( integrand , x=mu, axis=0) # integrate over mu axis (axis=0)
    return result[result!=0] # ensure no zero contributions after kcuts

def sigma_error(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ### dampsignal=True: will not contain instrumental effects in noise (thus errors) since it remains in signal
    P_obs_ = P_obs(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal)
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2,k_fg,dpix,dnu,kcuts = surveypars
    if tracer=='1': return P_obs_/np.sqrt(Nmodes(k,V_bin1))
    if tracer=='2': return P_obs_/np.sqrt(Nmodes(k,V_bin2))
    if tracer=='X':
        P_1 = P_obs(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer='1',dampsignal=dampsignal)
        P_2 = P_obs(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer='2',dampsignal=dampsignal)
        return np.sqrt( 1/(2*Nmodes(k,V_binX)) * (P_obs_**2 + P_1*P_2) )

def sigma_ell_error(ell,k_1d,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True):
    ''' Integrate error model over mu into multipole ell '''
    ### dampsignal=True: will not contain instrumental effects in noise (thus errors) since it remains in signal
    z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1,P_N2,k_fg,dpix,dnu,kcuts = surveypars
    mu_1d = np.linspace(-1,1,1000)
    k,mu = np.meshgrid(k_1d,mu_1d)
    integrand = sigma_error(k,mu,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal)**2 * Leg(ell)(mu)**2 * apply_kcuts(k,mu,kcuts)
    result = np.sqrt( (2*ell + 1)**2/2 * scipy.integrate.simpson( integrand , x=mu_1d, axis=0) ) # integrate over mu axis (axis=0)
    return result[result!=0] # ensure no zero contributions after kcuts

def Nmodes(k,V_bin):
    # Full mu-range effective count for each k-shell: 
    # N = (V k^2 dk / 8π^2) * ∫_{-1}^{1} dmu = V k^2 dk / 8π^2 * 2
    # Do not include dmu here; the mu integration is handled by scipy.integrate.simpson.
    dk = np.mean(np.diff(k[0,:]))
    return k**2*dk*V_bin / (8*np.pi**2) * 2

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

def B_grid(mu,k,z,cosmopars,dpix=None,dnu=None,novox=False):
    '''Top-hat damping due to sampling into (RA,Dec,nu) sky voxels, then regridding into
    larger Cartesian cells for Pk estimation. Assume Cartesian cells twice the size as sky voxels'''
    # dpix = angle size [degrees] along one side (commonly 0.3 or 0.5 degrees for MeerKLASS maps)
    # dnu = frequency resolution [MHz]: 0.210MHz (~0.133MHz) for MeerKAT L-band (UHF-band)
    # novox: set True if computing damping for galaxies or noise, in which case no voxelisation term is included (just gridding)
    # ----------------
    # Pixelisation:
    if dpix is None: B_pix = 1
    else:
        d_c = cosmo.D_com(z,cosmopars)
        s_pix = d_c * np.radians(dpix)
        k_perp = k*np.sqrt(1-mu**2)
        q = k_perp*s_pix/2
        B_pix = np.divide(np.sin(q),q,out=np.ones_like(q),where=q!=0.)
    # Channelisation:
    if dnu is None: B_chan = 1
    else:
        s_chan = c_km/cosmo.H(z) * (1+z)**2 * dnu/v_21cm
        k_para = k*mu
        q = k_para*s_chan/2
        B_chan = np.divide(np.sin(q),q,out=np.ones_like(q),where=q!=0.)
    # Gridding (apply additional step of discretiasation due to gridding):
    if dpix is None: B_xy = 1
    else:
        s_xy = s_pix*2 # cells twice the size of pixels for good sampling
        q = k_perp*s_xy/2
        B_xy = np.divide(np.sin(q),q,out=np.ones_like(q),where=q!=0.)
    if dnu is None: B_z = 1
    else:
        s_z = s_chan*2 # cells twice the size of channels for good sampling
        q = k_para*s_z/2
        B_z = np.divide(np.sin(q),q,out=np.ones_like(q),where=q!=0.)
    if novox==True: return B_xy * B_z # no pixelisation for galaxies (sampled straight to grid)
    else: return B_pix * B_chan * B_xy * B_z

def B_fg(mu,k,k_fg=0):
    if k_fg==0: return 1
    k_para = k*mu
    k_para[k_para==0] = 1e-30
    return 1 - np.exp( -(k_para/k_fg)**2 )

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
            P1D[i] = scipy.integrate.simpson(integrand, x=k_perp[mask], axis=0) # integrate over mu axis (axis=0)
    return kpara_centers, P1D


def Pk_noBAO(Pk,k,#kBAO=[0.03,0.3]
        ):
    kBAO=[k[1],k[-2]] # Use this as default instead

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
