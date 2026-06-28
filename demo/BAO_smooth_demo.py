import numpy as np
import matplotlib.pyplot as plt

################################################################
##### Update to pip install into environment:
import sys
sys.path.insert(1,'/Users/sadmin/Documents/MeerFish')
################################################################
################################################################

import cosmo
import survey
import model

### Survey parameters:
zminzmax = [0.8,1]
A_sky = 5000
z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1='MK_UHF',zminzmax=zminzmax,A_sky1=A_sky,t_tot=500,)
k_fg = 0 # explore this later
dpix,dnu = None,None
kcuts = None
surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar,k_fg,dpix,dnu,kcuts
dbeam,dsys,dphotoz = 0,0,0
nuispars = dbeam,dsys,dphotoz

### Cosmological parameters:
cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
Pmod = cosmo.MatterPk(z)

### k-bins, multipoles and tracer:
kmax = 0.2
k,kbins,kmin,kmax = model.get_kbins(z,zminzmax[0],zminzmax[1],A_sky=A_sky,kmax=kmax,Taruya=True)
ell = 0
tracer = '1'

### Obtain power spectrum model:
P_ell = model.P_ell(ell,k,Pmod,cosmopars,surveypars,nuispars,tracer)
P_ell_err = model.sigma_ell_error(ell,k,Pmod,cosmopars,surveypars,nuispars,tracer)
P_smoooth,f_BAO = model.Pk_noBAO(P_ell,k)

plt.errorbar(k,P_ell/P_smoooth,P_ell_err/P_smoooth)
plt.plot(k,P_smoooth/P_smoooth,color='black',ls='--')
plt.xlabel(r'$k [h/{\rm Mpc}]$')
plt.ylabel(r'$f_{\rm BAO} = P_\ell(k)\,/\,P_{\rm smooth}(k)$')
plt.savefig('plots/smoothedBAO_example.pdf')
plt.show()