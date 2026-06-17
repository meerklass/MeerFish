import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/Users/sadmin/Documents/MeerFish') # path to MeerFish code
import cosmo
import survey
import model
import fisher
import scipy

f_tobsloss = 0.5 # fraction of observation time loss (XLP assumed 0.5)
N_dish = 60 # number of operational dishes out of full 64 in array (XLP assumed 60)

## parameters to include in Fisher forecast:
theta_ids = [\
    #r'$\overline{T}_{\rm HI}$',\
    #r'$b_1$',\
    #r'$b_2$',\
    #r'$b^\phi_1$',\
    #r'$b^\phi_2$',\
    #r'$f$',\
    #r'$\alpha_\perp$',\
    #r'$\alpha_\parallel$',\
    r'$A_{\rm BAO}$',\
    #r'$f_{\rm NL}$'\
    ]

Survey1_arg = 'MK_UHF' # MeerKLASS UHF-band IM survey
Survey2_arg = 'DESI_LRG' # DESI LRG galaxies
t_obs = 800 * (1-f_tobsloss) # total MeerKLASS observation time (with losses then applied)
A_sky = 4000 # original sky area
A_skyX = 3000 # sky area overlapping with DESI galaxies

ZesVersion = True # use Ze's (XLP) models for HI bias and \bar{T}_HI

kmax = 0.1/0.7 # [h/Mpc] matching Ze's (XLP) cut - beyond this non-linear/gridding and instrumental effects dominate

### Choose which power spectrum multipoles:
ells = [0] # just monopole (default)

z = 0.8
deltaz = 0.2 # redshift width for each bin
zminzmax = [z-deltaz/2,z+deltaz/2] # use default

z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey1_arg,Survey2=Survey2_arg,zminzmax=zminzmax,A_sky1=A_sky,t_tot=t_obs,N_dish=N_dish,T_sys=model.Tsys(np.mean(zminzmax)))
k_fg = 0 # explore this later
dpix,dnu = None,None

kcuts = None
surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar,k_fg,dpix,dnu,kcuts

cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True,ZesVersion=ZesVersion) # set initial default cosmology
nuispars = 0,0,0 # ignore nuisance parameters throughout
Pmod = cosmo.MatterPk(z)
k,kbins,kmin,kmax = model.get_kbins(z,zminzmax[0],zminzmax[1],A_sky=A_skyX,kmax=kmax,Taruya=True)

### SNR from Fisher forecast method:
F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='1')
C = fisher.FisherInverse(F)
BAO_SNR = 1/np.sqrt(C[-1][-1])
print(BAO_SNR)

### SNR added in quadrature method:
P_X = model.P_ell(0,k,Pmod,cosmopars,surveypars,nuispars,'1')
P_err = model.sigma_ell_error(0,k,Pmod,cosmopars,surveypars,nuispars,'1')
P_smooth,f_BAO = model.Pk_noBAO(P_X,k)
f_err = P_err/P_smooth
BAO_SNR = np.sqrt( np.sum((f_BAO/f_err)**2) )
print(BAO_SNR)
BAO_SNR = np.sqrt(np.sum((P_X - P_smooth)**2 / P_err**2))
print(BAO_SNR)

F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='1',Bernal=False)
C = fisher.FisherInverse(F)
BAO_SNR = 1/np.sqrt(C[-1][-1])
print(BAO_SNR)

F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='1',Bernal=True)
C = fisher.FisherInverse(F)
BAO_SNR = 1/np.sqrt(C[-1][-1])
print(BAO_SNR)

