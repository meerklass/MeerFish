import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.interpolate import interp1d
import model as meerfish_model
c_km = 299792.458 #km/s

def SetCosmology(builtincosmo='Planck18',z=0,return_cosmopars=False):
    import camb
    from camb import model, initialpower
    global H_0
    global h
    global D_z # Growth function D(z)
    global Om0
    global Ob0
    global n_s
    global A_s
    global delta_c
    if builtincosmo=='WMAP1':
        H_0 = 73 # [km/(Mpc s)]
        h = H_0/100
        Om0 = 0.25
        Ob0 = 0.045 # Omega_b
        n_s = 0.99
    if builtincosmo=='Planck15':
        H_0 = 67.7 # [km/(Mpc s)]
        h = H_0/100
        Om0 = 0.307
        Ob0 = 0.0486 # Omega_b
        n_s = 0.968
    if builtincosmo=='Planck18':
        H_0 = 67.4 # [km/(Mpc s)]
        h = H_0/100
        Om0 = 0.315 # Omega_m
        Ob0 = 0.0489 # Omega_b
        n_s = 0.965
    A_s = 2.14e-9 # Scalar amplitude
    D_z = D(z) # Growth Function (normalised to unity for z=0)
    delta_c = 1.686
    MatterPk(z,1e-4,1e0,NonLinear=False) # Use to set global transfer function T
    if return_cosmopars==True:
        Tbar1 = meerfish_model.Tbar(z,meerfish_model.OmegaHI(z))
        Tbar2 = 1
        b1 = meerfish_model.b_HI(z)
        b2 = 1 + z
        bphi1 = b_phi_universality(b1)
        bphi2 = b_phi_universality(b2)
        f_ = f(z)
        a_perp = 1
        a_para = 1
        A_BAO = 1
        f_NL = 0
        cosmopars = np.array([Tbar1,Tbar2,b1,b2,bphi1,bphi2,f_,a_perp,a_para,A_BAO,f_NL])
        return cosmopars

def f(z):
    gamma = 0.545
    return Omega_M(z)**gamma

def b_phi_universality(b1):
    ''' For given linear bias b1, return the PNG bias assuming b_phi_universality'''
    return 2*delta_c*(b1-1) # assume universality until better model found

def b_phi_HI(z):
    '''Interpolated from Alex Barriera paper: https://iopscience.iop.org/article/10.1088/1475-7516/2022/04/057/pdf'''
    #### Code for finding polynomial coeficients: #####
    '''
    z = np.array([0,0.5,1,2,3])
    b_phi = np.array([-1.7, -0.44, 0.47, 2.39, 3.82]) # From TNG100
    #b_phi = np.array([-1.76, -1.40, -0.26, 2.08, 1.98]) # From TNG300
    coef = np.polyfit(z, b_phi,2)
    A,B,C = coef[2],coef[1],coef[0]
    '''
    ###################################################
    A,B,C = -1.673166311300638,2.4020042643923234,-0.18997867803837928
    return A + B*z + C*z**2

def E(z):
    return np.sqrt( 1 - Om0 + Om0*(1+z)**3 )

def H(z):
    return E(z)*H_0

def Omega_M(z):
    return H_0**2*Om0*(1+z)**3 / H(z)**2

def Omega_b(z=0):
    if z!=0: print('\nError: Cosmotools needs evoloution for Omega_b(z)!')
    return Ob0

def D_com(z,cosmopars=None):
    if cosmopars is not None:
        Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A,f_NL = cosmopars

        a_para = 1

        H_0_fid = np.copy(H_0)
        H_0_new = H_0_fid/a_para
        #Comoving distance [Mpc/h]
        func = lambda z: (c_km/H_0_new)/E(z)
        #h_new = H_0_new/100
        #return scipy.integrate.romberg(func,0,z) * h_new
        #### NOT DIALATING little h ######
        h = H_0_fid/100
        return scipy.integrate.romberg(func,0,z) * h
        ####################################

    #Comoving distance [Mpc/h]
    func = lambda z: (c_km/H_0)/E(z)
    h = H_0/100
    return scipy.integrate.romberg(func,0,z) * h

def D_ang(A,z):
    '''Compute comoiving transverse distance for a survey with area A at redshift z'''
    from astropy.cosmology import Planck18 as cosmo_astropy
    d_tr = cosmo_astropy.comoving_transverse_distance(z).value*h  # [Mpc/h]
    area_sr = (A / 41253) * (4 * np.pi) # [steradians]
    A_transverse = d_tr**2 * area_sr
    return np.sqrt(A_transverse) # [Mpc/h]

def D_A(z):
    return D_com(z) / (1+z)

def Deltab(z,k,f_NL,b_HI):
    #Scale dependent modification of the Gaussian clustering bias
    # Eq 13 of https://arxiv.org/pdf/1507.03550.pdf
    T_k = np.ones(np.shape(k)) # Use to avoid evaluating T(k) outside interpolation
                               #   range. Uses convention T(k->0)=1
    T_k[k>kmin_interp] = T(k[k>kmin_interp]) # Evalutate transfer function for all k>0
    k[k==0] = 1e-30 # Avoid divive by zero error in final return statement
    return 3*f_NL*( (b_HI-1)*Om0*H_0**2*delta_c ) / (c_km**2*k**2*T_k*D_z)

def M(k,z):
    ## PNG parameter as defined after eq2.1 in https://arxiv.org/pdf/2302.09066.pdf
    T_k = np.ones(np.shape(k)) # Use to avoid evaluating T(k) outside interpolation
                               #   range. Uses convention T(k->0)=1
    T_k[k>kmin_interp] = T(k[k>kmin_interp]) # Evalutate transfer function for all k>0
    return 2/3 * c_km**2 * k**2 * T_k * D_z / (Om0 * H_0**2)
    #return 2/3 * k**2 * T_k / (Om0 * H_0**2) # version from eq1.5 https://arxiv.org/pdf/2006.09368.pdf

def D(z):
    #Growth parameter - obtained from eq90 in:
    #   https://www.astro.rug.nl/~weygaert/tim1publication/lss2009/lss2009.linperturb.pdf
    integrand = lambda zi: (1+zi)/(H(zi)**3)
    D_0 = 5/2 * Om0 * H_0**2 * H(0) * integrate.quad(integrand, 0, 1e3)[0]
    if z==0: return D_0 # Growth factor normalisation factor
    D_z = 5/2 * Om0 * H_0**2 * H(z) * integrate.quad(integrand, z, 1e3)[0]
    return D_z / D_0 # Normalise such that D(z=0) = 1

def sigma_8(z=None):
    if z is None: # Extract sigma_8 at CAMB redshit using defined global results from MatterPk function
        return results.get_sigma8()[0]
    else: # avoid this if possible because runs camb each time
        import camb
        Oc0 = Om0 - Ob0 # Omega_c
        #Set up the fiducial cosmology
        pars = camb.CAMBparams()
        #Set cosmology
        pars.set_cosmology(H0=H_0,ombh2=Ob0*h**2,omch2=Oc0*h**2,omk=0,mnu=0)
        pars.set_dark_energy() #LCDM (default)
        pars.InitPower.set_params(ns=n_s, r=0, As=A_s)
        results_loc = camb.get_results(pars)
        #Get matter power spectrum at some redshift
        pars.set_matter_power(redshifts=[z])
        results_loc.calc_power_spectra(pars)
        return results_loc.get_sigma8()[0]

def MatterPk(z,kmin=1e-5,kmax=10,NonLinear=False):
    '''
    Use pycamb to generate model power spectrum at redshift z
    '''
    import camb
    from camb import model, initialpower
    # Declare minium k value for avoiding interpolating outside this value
    global kmin_interp
    kmin_interp = kmin
    Oc0 = Om0 - Ob0 # Omega_c
    #Set up the fiducial cosmology
    pars = camb.CAMBparams()
    #Set cosmology
    pars.set_cosmology(H0=H_0,ombh2=Ob0*h**2,omch2=Oc0*h**2,omk=0,mnu=0)
    pars.set_dark_energy() #LCDM (default)
    pars.InitPower.set_params(ns=n_s, r=0, As=A_s)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    #Calculate results for these parameters
    global results
    results = camb.get_results(pars)
    #Get matter power spectrum at some redshift
    pars.set_matter_power(redshifts=[z], kmax=kmax)
    if NonLinear==False: pars.NonLinear = model.NonLinear_none
    if NonLinear==True: pars.NonLinear = model.NonLinear_both # Uses HaloFit
    results.calc_power_spectra(pars)
    k, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = 200)
    # Define global transfer function to be called in other functions:
    trans = results.get_matter_transfer_data()
    k_trans = trans.transfer_data[0,:,0] #get kh - the values of k/h at which transfer function is calculated
    transfer_func = trans.transfer_data[model.Transfer_cdm-1,:,0]
    transfer_func = transfer_func/np.max(transfer_func)
    global T
    T = interp1d(k_trans, transfer_func) # Transfer function - set to global variable
    return interp1d(k,pk[0],kind='cubic')


###########################  CHECK!  ##########################################
# ----------------------
# Modified gravity approximations of fsigma8 parameterised  by \mu following ChatGPT
# ----------------------
###############################################################################
# ----------------------
# Background functions
# ----------------------
def E_of_a(a):
    return np.sqrt(Om0 * a**-3 + Omega_L0)

def dlnH_dlnA(a):
    E2 = Om0 * a**-3 + Omega_L0
    return -0.5 * (3 * Om0 * a**-3) / E2

def Omega_m_of_a(a):
    E2 = Om0 * a**-3 + Omega_L0
    return (Om0 * a**-3) / E2

def Omega_DE_ratio(a):
    return 1.0 / (E_of_a(a)**2)

def mu_of_a(a, mu0):
    return 1.0 + mu0 * Omega_DE_ratio(a)
# ----------------------
# Growth ODE integration
# ----------------------
def fsigma8_mu0(z_array, mu0, a_init=1e-3, return_D=False):
    global Omega_L0
    Omega_L0 = 1.0 - Om0  # flat universe

    a_grid = np.logspace(np.log10(a_init), 0.0, 4000)
    y_grid = np.log(a_grid)
    f = np.zeros_like(a_grid)
    f[0] = 1.0  # initial condition in matter domination

    def RHS(ff, aa):
        return 1.5 * Omega_m_of_a(aa) * mu_of_a(aa, mu0) - ff**2 - (2.0 + dlnH_dlnA(aa)) * ff

    for i in range(len(a_grid)-1):
        a = a_grid[i]
        h = y_grid[i+1] - y_grid[i]
        k1 = RHS(f[i], a)
        a_mid = np.sqrt(a * a_grid[i+1])
        f_mid = f[i] + 0.5 * h * k1
        k2 = RHS(f_mid, a_mid)
        f[i+1] = f[i] + h * k2

    lnD = np.cumsum(f * np.gradient(y_grid))
    D = np.exp(lnD)
    # --- Early-time normalization (so that D ~ a at early times) ### CHANGED
    norm = D[0] / a_grid[0]
    D /= norm

    def interp(vec, a_vals, a_query):
        idx = np.searchsorted(a_vals, a_query)
        idx = np.clip(idx, 1, len(a_vals)-1)
        x0, x1 = a_vals[idx-1], a_vals[idx]
        y0, y1 = vec[idx-1], vec[idx]
        t = (a_query - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    if return_D:  # just return growth factor values at requested z’s
        return np.array([interp(D, a_grid, 1.0/(1.0+z)) for z in z_array])

    f_interp = lambda a: interp(f, a_grid, a)
    D_interp = lambda a: interp(D, a_grid, a)

    f_sigma8 = []
    for z in z_array:
        a = 1.0 / (1.0 + z)
        f_val = f_interp(a)
        D_val = D_interp(a)
        sigma8_z = sigma_8(z)
        f_sigma8.append(f_val * sigma8_z)
    return np.array(f_sigma8)
###############################################################################
###############################################################################
###############################################################################
