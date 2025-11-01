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
dbeam,dsys,dphotoz = 0,0,0
nuispars = dbeam,dsys,dphotoz

### Cosmological parameters:
Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
cosmopars = np.array([Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL])
Pmod = cosmo.MatterPk(z)

### k-bins, multipoles and tracer:
kmin = 0.02
kmax = 0.3
kbins = np.linspace(kmin,kmax,200)
k = (kbins[1:] + kbins[:-1])/2 #centre of k bins
ell = 0
tracer = '1'

### Obtain power spectrum model:
import model
P_ell = model.P_ell(ell,k,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True)
plt.plot(k,P_ell,color='black',ls='--')
P_ell_obs = model.P_ell_obs(ell,k,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True)
plt.plot(k,P_ell_obs,color='tab:blue')
P_ell_err = model.sigma_ell_error(ell,k,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True)
plt.fill_between(k,(P_ell_obs-P_ell_err),(P_ell_obs+P_ell_err),alpha=0.6,color='tab:blue')
plt.loglog()
plt.figure()

### Observational effects on quadrupole:
ell = 2
tracer  = 'X'
cosmopars_nogrowth = np.array([Tbar1,Tbar2,b1,b2,bphi1,bphi2,0,a_perp,a_para,A_BAO,f_NL])
surveypars_nobeam = z,V_bin1,V_bin2,V_binX,0,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
Pobs_nogrowth = model.P_ell_obs(ell,k,Pmod,cosmopars_nogrowth,surveypars_nobeam,nuispars,tracer,dampsignal=True)
sig_err = model.sigma_ell_error(ell,k,Pmod,cosmopars_nogrowth,surveypars_nobeam,nuispars,tracer=tracer,dampsignal=True)
plt.plot(k,k**2*Pobs_nogrowth,color='black',label='no growth, no beam')
plt.fill_between(k,k**2*(Pobs_nogrowth-sig_err),k**2*(Pobs_nogrowth+sig_err),alpha=0.6,color='black')

Pobs_nogrowth = model.P_ell_obs(ell,k,Pmod,cosmopars_nogrowth,surveypars,nuispars,tracer,dampsignal=True)
sig_err = model.sigma_ell_error(ell,k,Pmod,cosmopars_nogrowth,surveypars,nuispars,tracer=tracer,dampsignal=True)
plt.plot(k,k**2*Pobs_nogrowth,color='tab:red',label='no growth, with beam')
plt.fill_between(k,k**2*(Pobs_nogrowth-sig_err),k**2*(Pobs_nogrowth+sig_err),alpha=0.6,color='tab:red')

Pobs = model.P_ell_obs(ell,k,Pmod,cosmopars,surveypars,nuispars,tracer,dampsignal=True)
sig_err = model.sigma_ell_error(ell,k,Pmod,cosmopars,surveypars,nuispars,tracer=tracer,dampsignal=True)
plt.plot(k,k**2*Pobs,color='tab:blue',label='with growth, with beam')
plt.fill_between(k,k**2*(Pobs-sig_err),k**2*(Pobs+sig_err),alpha=0.6,color='tab:blue')
plt.legend(fontsize=12)
plt.xlim(k[0],k[-1])
plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
plt.ylabel(r'$k^2\,P_2(k)\,[h\,{\rm Mpc}\,{\rm mK}]$')
plt.show()
exit()

### Effect of t_obs on quadrupole:
plt.figure(figsize=(16,8))
ell = 2
tracer  = 'X'
t_tots = [2,20,50,100]
for i in range(len(t_tots)):
    P_N1_i = model.P_N(z,A_sky1,t_tots[i],N_dish,theta_FWHM=theta_FWHM1)
    surveypars_i = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N1_i,1/nbar
    plt.subplot(141+i)
    Pobs_nogrowth = model.P_ell_obs(ell,k,Pmod,cosmopars_nogrowth,surveypars_i,tracer)
    sig_err = model.sigma_ell_error(ell,k,Pmod,cosmopars_nogrowth,surveypars_i,tracer=tracer)
    plt.plot(k,k**2*Pobs_nogrowth,color='tab:red',label='no growth')
    plt.fill_between(k,k**2*(Pobs_nogrowth-sig_err),k**2*(Pobs_nogrowth+sig_err),alpha=0.6,color='tab:red')

    Pobs = model.P_ell_obs(ell,k,Pmod,cosmopars,surveypars_i,tracer)
    sig_err = model.sigma_ell_error(ell,k,Pmod,cosmopars,surveypars_i,tracer=tracer)
    plt.plot(k,k**2*Pobs,color='tab:blue',label='with growth')
    plt.fill_between(k,k**2*(Pobs-sig_err),k**2*(Pobs+sig_err),alpha=0.6,color='tab:blue')

    plt.legend(fontsize=12)
    plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
    if i==0: plt.ylabel(r'$k^2\,P_2(k)\,[h\,{\rm Mpc}\,{\rm mK}]$')
    plt.title(r'Obs time = $%s$hrs'%t_tots[i],fontsize=18)
    plt.xlim(k[0],k[-1])
plt.figure()
#exit()

### Fisher forecast:
import fisher
theta_ids = [\
#r'$\overline{T}_{\rm HI}$',\
r'$b_1$',\
#r'$b_2$',\
#r'$b^\phi_1$',\
#r'$b^\phi_2$',\
r'$f$',\
#r'$\alpha_\perp$',\
#r'$\alpha_\parallel$',\
#r'$A_{\rm BAO}$',\
#r'$f_{\rm NL}$'\
]

ells = [0,2]
theta = model.get_param_vals(theta_ids,z,cosmopars)
F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,ells,tracer='1')
fisher.CornerPlot(F,theta,theta_ids)
plt.show()
