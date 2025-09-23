import numpy as np
import matplotlib.pyplot as plt

################################################################
##### Update to pip install into environment:
import sys
sys.path.insert(1,'/Users/user/Documents/MeerFish')
################################################################
################################################################

import cosmo
import survey

### Survey parameters:
z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params()
surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar

### Cosmological parameters:
Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
cosmopars = np.array([Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL])
Pmod = cosmo.MatterPk(z)

### k-bins:
kmin = 0.02
kmax = 0.3
kbins = np.linspace(kmin,kmax,200)
k = (kbins[1:] + kbins[:-1])/2 #centre of k bins

### Test changing beam:
ells = [0,2]
tracer  = '1'
import model
import fisher
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
theta = model.get_param_vals(theta_ids,z,cosmopars)

for th in theta_ids:
    fisher.check_stencil_convergence(th, ells, k, tracer,Pmod,cosmopars, surveypars)
