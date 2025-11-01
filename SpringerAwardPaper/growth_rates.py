import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/Users/user/Documents/MeerFish')
import cosmo
import model
import fisher
import survey

### For nice plots:
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import mpl_style
plt.style.use(mpl_style.style1)

epsilon = 0.5 # fraction of total observation time kept (i.e. not lost to RFI)
f_tobsloss = 1-epsilon

ells = [0,2,4]
theta_ids = [\
    #r'$\overline{T}_{\rm HI}$',\
    r'$b_1$',\
    #r'$b_2$',\
    #r'$b^\phi_1$',\
    #r'$b^\phi_2$',\
    r'$f$',\
    r'$\alpha_\perp$',\
    r'$\alpha_\parallel$',\
    #r'$A_{\rm BAO}$',\
    r'$f_{\rm NL}$'\
    ]

########################################################################################################
#### Growth with redshift:
########################################################################################################
#'''
zbins = np.array([0.4,0.6,0.8,1.0,1.2,1.45])
f,f_err,sigma_8,zs = [],[],[],[]
for i in range(len(zbins)-1):
    ### Full MeerKLASS survey at each z-bin:
    Survey = 'MK_UHF'
    zminzmax = [zbins[i],zbins[i+1]]
    A_sky = None # None is default
    z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey,A_sky1=A_sky,zminzmax=zminzmax,f_tobsloss=epsilon)
    surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
    dbeam,dsys,dphotoz = 0,0,0
    nuispars = dbeam,dsys,dphotoz

    ### Cosmological parameters and kbins:
    cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
    Pmod = cosmo.MatterPk(z)
    k,kbins,kmin,kmax = model.get_kbins(z,zmin1,zmax1,A_sky1)

    theta = model.get_param_vals(theta_ids,z,cosmopars)
    F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='1')
    C = fisher.FisherInverse(F)
    f.append(theta[1])
    f_err.append(np.sqrt(C[1,1]))
    sigma_8.append(cosmo.sigma_8())
    zs.append(z)

    '''
    #### Hack test to see contribution from non-linear modes:
    print((np.array(f_err)/np.array(f))*100) # percent constraints
    kcut = 0.15 # assume all modes above this non-linear and cut
    k = k[k<kcut]
    F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,ells,tracer='1')
    C = fisher.FisherInverse(F)
    print((np.sqrt(C[1,1])/theta[1])*100) # percent constraints
    print( (np.sqrt(C[1,1])/theta[1]) / (np.array(f_err)/np.array(f)) ) # gain from non-linear mode inclusion
    exit()
    '''

f,f_err,sigma_8,zs = np.array(f),np.array(f_err),np.array(sigma_8),np.array(zs)
np.save('/Users/user/Documents/MeerFish/SpringerAwardPaper/data/f_z.npy',[f,f_err,sigma_8,zs])
#'''
plt.figure(figsize=(6,4))
f,f_err,sigma_8,zs = np.load('/Users/user/Documents/MeerFish/SpringerAwardPaper/data/f_z.npy',allow_pickle=True)
print((f_err/f)*100) # percent constraints
plt.errorbar(zs,f*sigma_8,f_err*sigma_8,ls='none',marker='o',markersize=6,label='MeerKLASS')

'''
#eBOSS constraints (from Table 9 in: https://arxiv.org/pdf/2007.09013):
z_g = np.array([0.74,0.847,1.478])
fsig8_g = np.array([0.5,0.52,0.30])
ferr_g = np.array([0.11,0.10,0.13])
'''
#
#eBOSS constraints (from Table 3 in: https://arxiv.org/pdf/2007.08991):
z_g = np.array([0.15,0.38,0.51,0.70,0.85,1.48])
fsig8_g = np.array([0.53,0.5,0.455,0.448,0.315,0.462])
ferr_g = np.array([0.16,0.047,0.039,0.043,0.095,0.045])
plt.errorbar(z_g,fsig8_g,ferr_g,ls='none',marker='x',color='tab:orange',label='eBOSS',elinewidth=1,markersize=4,alpha=0.4,zorder=-10)

# DESI (from Table 9, Table 1 and Fig 14 in https://arxiv.org/pdf/2411.12021):
z_g = np.array([0.295,0.510,0.706,0.919,1.317,1.491])
dfsig8_g = np.array([0.8,1.09,1.05,0.96,0.95,1.16])
dferr_g_low = np.array([0.2,0.14,0.12,0.10,0.08,0.12])
dferr_g_upp = np.array([0.2,0.12,0.12,0.11,0.11,0.12])
'''
# DESI PS+BS (from Table 1 and Table 6 in https://arxiv.org/pdf/2503.09714):
z_g = np.array([0.51,0.706,0.920,1.491])
dfsig8_g = np.array([1.087,1.047,1.029,1.113])
dferr_g_low = np.array([0.075,0.095,0.062,0.126])
dferr_g_upp = np.array([0.133,0.099,0.137,0.086])
'''
# loop over redshifts to obtain fiducial f(z) to report DESI results in absolute terms:
for i in range(len(z_g)):
    cosmo.SetCosmology(z=z_g[i])
    Pmod = cosmo.MatterPk(z_g[i])
    dfsig8_g[i] = cosmo.f(z_g[i])*cosmo.sigma_8() * dfsig8_g[i]
    dferr_g_low[i] = cosmo.f(z_g[i])*cosmo.sigma_8() * dferr_g_low[i]
    dferr_g_upp[i] = cosmo.f(z_g[i])*cosmo.sigma_8() * dferr_g_upp[i]
asymmetric_error = np.array(list(zip(dferr_g_low, dferr_g_upp))).T
plt.errorbar(z_g,dfsig8_g,asymmetric_error,ls='none',marker='s',color='tab:green',label='DESI',elinewidth=1,markersize=3,alpha=0.4,zorder=-10)

zs = np.linspace(0.134,1.521,40)

'''
### MG models parameterised by \gamma:
gamma_GR = 0.545
gamma_DGP = 0.68
gamma_fR = 0.42
w = 0.856
gamma_w =  0.55 + 0.02*(1 + w)
sigma_8 = []
for i in range(len(zs)):
    print(i)
    sigma_8.append(cosmo.sigma_8(zs[i]))
plt.plot(zs,sigma_8*cosmo.Omega_M(zs)**gamma_GR,label='GR',color='gray',lw=1.5)
plt.plot(zs,sigma_8*cosmo.Omega_M(zs)**gamma_fR,ls=':',label=r'$f(R)$',color='gray',lw=1)
plt.plot(zs,sigma_8*cosmo.Omega_M(zs)**gamma_DGP,ls='--',label='DGP',color='gray',lw=1)
'''

### MG models parameterised by mu0 (more generic) - see https://arxiv.org/pdf/2411.12021:
plt.plot(zs,cosmo.fsigma8_mu0(zs,0),label='GR',color='gray',lw=1.5)
plt.plot(zs,cosmo.fsigma8_mu0(zs,0.5),ls='--',label=r'$\mu_0\,{=}\,{+}0.5$',color='gray',lw=1)
plt.plot(zs,cosmo.fsigma8_mu0(zs,-0.5),ls=':',label=r'$\mu_0\,{=}\,{-}0.5$',color='gray',lw=1)

plt.xlim(0.135,1.52)
plt.ylim(0.3,0.56)
plt.legend(loc='lower left',ncol=2,fontsize=16,frameon=True,framealpha=0.7,borderaxespad=0.2,columnspacing=1,handletextpad=0.4,handlelength=1)
plt.xlabel(r'$z$')
plt.ylabel(r'$f\,\sigma_8(z)$')
plt.savefig('/Users/user/Documents/MeerFish/SpringerAwardPaper/plots/growth_rates.pdf',bbox_inches='tight') # Spinger Awards
plt.show()
