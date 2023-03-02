import numpy as np
import cosmo
import scipy
from scipy import integrate
v_21cm = 1420.405751#MHz

def Red2Freq(z):
    # Convert redshift to frequency for HI emission (freq in MHz)
    return v_21cm / (1+z)

def Freq2Red(v):
    # Convert frequency to redshift for HI emission (freq in MHz)
    return (v_21cm/v) - 1

def b_HI(z):
    ''' HI linear bias '''
    '''
    Use 6 values for HI bias at redshifts 0 to 5 found in Table 5 of
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
    '''
    Use 6 values for HI shot noise at redshifts 0 to 5 found in Table 5 of
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
    if R_beam==0: return 1
    return np.exp( -(1-mu**2)*k**2*R_beam**2/2 )

def P_HI(k,mu,z,Pmod,cosmopars,surveypars):

    Omega_HI,b_HI,f,bphiHI,f_NL = cosmopars
    zmin,zmax,R_beam,A_sky,t_tot,N_dish = surveypars
    return Tbar(z,Omega_HI)**2 * (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) * B_beam(mu,k,R_beam)**2

def P_HI_obs(k,mu,z,Pmod,cosmopars,surveypars):

    Omega_HI,b_HI,f,bphiHI,f_NL = cosmopars
    zmin,zmax,R_beam,A_sky,t_tot,N_dish = surveypars
    return ( Tbar(z,Omega_HI)**2 * ( (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k,z)**(-1))**2 * Pmod(k) + P_SN(z) ) ) * B_beam(mu,k,R_beam)**2 + P_N(z,zmin,zmax,A_sky,t_tot,N_dish)

def Pk(Pmod,kbins,z,theta,surveypars=None,galcross=False):
    ''' HI signal only part of the power spectrum '''

    if galcross==False:
        Omega_HI,b_HI,f,bphiHI,f_NL = theta
    if galcross==True:
        Omega_HI,b_HI,b_g,f,bphiHI,bphig,f_NL, = theta

    if surveypars is None: t_tot,zmin,zmax,R_beam,A_sky,N_dish = None,0,0,0,0,0
    else: zmin,zmax,R_beam,A_sky,t_tot,N_dish = surveypars

    if galcross==False:
        Pk_int = lambda mu: Tbar(z,Omega_HI)**2 * (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k_i,z)**(-1))**2 * Pmod(k_i) * B_beam(mu,k_i,R_beam)**2
        Pk_int_obs = lambda mu: ( Tbar(z,Omega_HI)**2 * ( (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k_i,z)**(-1))**2 * Pmod(k_i) + P_SN(z) ) ) * B_beam(mu,k_i,R_beam)**2 + P_N(z,zmin,zmax,A_sky,t_tot,N_dish)
    if galcross==True:
        ##### REVISE #######
        Pk_int = lambda mu: Tbar1*Tbar2 * b1*b2*( r + (beta1 + beta2)*mu**2 + beta1*beta2*mu**4 ) / (1 + (k_i*mu*sigv/H_0)**2) * Pmod(k_i) * B_beam(mu,k_i,R_beam1) * B_beam(mu,k_i,R_beam2) + P_N
        ####################
    pkmod = np.zeros(len(k))
    pkmod_obs = np.zeros(len(k))
    nmodes = np.zeros(len(k))
    for i in range(len(k)):
        k_i = k[i]
        pkmod[i] = scipy.integrate.quad(Pk_int, 0, 1)[0]
        pkmod_obs[i] = scipy.integrate.quad(Pk_int_obs, 0, 1)[0]
        #nmodes[i] = 1 / (2*np.pi)**3 * (lx*ly*lz) * (4*np.pi*k_i**2*deltak[i]) # Based on eq14 in https://arxiv.org/pdf/1509.03286.pdf
        nmodes = 1
    return pkmod,pkmod_obs,k,nmodes
