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

def B_beam(mu,k,R_beam):
    if R_beam==0: return 1
    return np.exp( -(1-mu**2)*k**2*R_beam**2/2 )

def P_HI(Pmod,kbins,z,theta,R_beam=0,sig_N=0,galcross=False):
    k = (kbins[1:] + kbins[:-1])/2
    deltak = [kbins[i]-kbins[i-1] for i in range(1,len(kbins))]
    if galcross==False:
        Omega_HI,b_HI,f,bphiHI,f_NL = theta
        beta_HI = f/b_HI
    if galcross==True:
        Omega_HI,b_HI,b_g,f,bphiHI,bphig,f_NL, = theta
        beta_HI,beta_g = f/b_HI,f/b_g
    if sig_N!=0: P_N = sig_N**2 * (lx*ly*lz)/(nx*ny*nz) # noise term
    else: P_N = 0

    if galcross==False:
        Pk_int = lambda mu: (Tbar(z,Omega_HI)**2 * (b_HI + f*mu**2 + bphiHI*f_NL*cosmo.M(k_i,z)**(-1))**2 * Pmod(k_i) + P_SN(z)) * B_beam(mu,k_i,R_beam)**2 + P_N
    if galcross==True:
        ##### REVISE #######
        Pk_int = lambda mu: Tbar1*Tbar2 * b1*b2*( r + (beta1 + beta2)*mu**2 + beta1*beta2*mu**4 ) / (1 + (k_i*mu*sigv/H_0)**2) * Pmod(k_i) * B_beam(mu,k_i,R_beam1) * B_beam(mu,k_i,R_beam2) + P_N
        ####################
    pkmod = np.zeros(len(k))
    nmodes = np.zeros(len(k))
    for i in range(len(k)):
        k_i = k[i]
        pkmod[i] = scipy.integrate.quad(Pk_int, 0, 1)[0]
        #nmodes[i] = 1 / (2*np.pi)**3 * (lx*ly*lz) * (4*np.pi*k_i**2*deltak[i]) # Based on eq14 in https://arxiv.org/pdf/1509.03286.pdf
        nmodes = 1
    return pkmod,k,nmodes
