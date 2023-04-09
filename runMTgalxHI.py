import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
import cosmo
import model
import survey
import fisher

### Determine survey:
#Survey = 'MK_2019' # MeerKAT 2019 pilot survey in L-band
#Survey = 'MK_LB' # MeerKLASS L-band
Survey1 = 'MK_UHF' # MeerKLASS UHF-band
#Survey = 'SKA'

Survey2 = 'DESI_ELG'
z,zmin,zmax,R_beam,A_sky,t_tot,N_dish,nbar,V_bin = survey.params(Survey1,Survey2)

### k-bins:
kmin = np.pi/V_bin**(1/3) ### From Tayura https://arxiv.org/pdf/1101.4723.pdf (after eq8)
kmax = 0.4
kbins = np.arange(kmin,kmax,kmin)
k = (kbins[1:] + kbins[:-1])/2 #centre of k bins

### Cosmological/survey parameters:
Pmod = cosmo.MatterPk(z,kmin=kmin-kmin/1000)
Omega_HI = model.OmegaHI(z)
Tbar = model.Tbar(z,Omega_HI)
b_HI = model.b_HI(z)
f = cosmo.f(z)
D_A = cosmo.D_A(z)
H = cosmo.H(z)
A = 1
bphiHI = cosmo.b_phi_universality(b_HI)
f_NL = 0

b_g = 2
#b_g = b_HI
bphig = cosmo.b_phi_universality(b_g)

cosmopars = [Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL]
surveypars = [zmin,zmax,R_beam,A_sky,t_tot,N_dish,nbar]

ell = 0
P_HI = model.P_ell(ell,k,z,Pmod,cosmopars,surveypars,tracer='HI')
P_g = model.P_ell(ell,k,z,Pmod,cosmopars,surveypars,tracer='g')
P_gHI = model.P_ell(ell,k,z,Pmod,cosmopars,surveypars,tracer='X')
plt.plot(k,P_HI,label=r'$P_{{\rm HI},%s}$'%ell)
plt.plot(k,P_g,label=r'$P_{{\rm g},%s}$'%ell)
plt.plot(k,P_gHI,label=r'$P_{{\rm HI,g},%s}$'%ell)
plt.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=[0.5,0.98])
plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
plt.ylabel(r'$P_{{\rm HI,g},\ell}(k)\,[{\rm mK}\, h^{-3}\,{\rm Mpc}^{3}]$')
plt.loglog()
plt.figure()
#exit()

#### Check BAO-wiggles only: ######
'''
ell = 0
z = 0.8
deltaz = 0.2
zmin,zmax = z-deltaz/2,z+deltaz/2
### Recalcuate cosmology and redshift dependent params:
cosmo.SetCosmology(z=z)
Omega_HI = model.OmegaHI(z)
Tbar = model.Tbar(z,Omega_HI)
b_HI = model.b_HI(z)
f = cosmo.f(z)
D_A = cosmo.D_A(z)
H = cosmo.H(z)
bphiHI = cosmo.b_phi_universality(b_HI)
cosmopars = [Omega_HI,b_HI,b_g,f,D_A,H,A,bphiHI,bphig,f_NL]

A_sky = 10000
V_bin = survey.Vsur(zmin,zmax,A_sky)

surveypars = [zmin,zmax,R_beam,A_sky,t_tot,N_dish,nbar]
kmin = np.pi/V_bin**(1/3) ### From Tayura https://arxiv.org/pdf/1101.4723.pdf (after eq8)
kbins = np.arange(kmin,kmax,kmin)
k = (kbins[1:] + kbins[:-1])/2 #centre of k bins
deltak = k[1]-k[0]

Pmod = cosmo.MatterPk(z,kmin=kmin-kmin/1000)
P_HI_obs = model.P_ell_obs(ell,k,z,Pmod,cosmopars,surveypars,tracer='HI')
P_g_obs = model.P_ell_obs(ell,k,z,Pmod,cosmopars,surveypars,tracer='g')

P_obs = model.P_ell_obs(ell,k,z,Pmod,cosmopars,surveypars,tracer='HI')
sig_err = np.sqrt(2*(2*np.pi)**3/V_bin*1/(4*np.pi*k**2*deltak)) * P_obs
if ell==0: P_obs -= model.P_N(z,zmin,zmax,A_sky,t_tot,N_dish) # subtract noise

P_obs = model.P_ell_obs(ell,k,z,Pmod,cosmopars,surveypars,tracer='X')
sig_err = np.sqrt((2*np.pi)**3/V_bin*1/(4*np.pi*k**2*deltak)) * np.sqrt(P_obs**2 + P_HI_obs*P_g_obs)

P_smooth,f_BAO = model.Pk_noBAO(P_obs,k)

plt.plot(k,f_BAO)
plt.axhline(0,color='grey',lw=0.5,zorder=-1)
plt.fill_between(k, f_BAO-sig_err/P_smooth, f_BAO+sig_err/P_smooth,alpha=0.4,zorder=-2)
plt.xscale('log')
plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$')
plt.ylabel(r'$f_{\rm BAO}$')
plt.suptitle(r'Area=%s, z=%s, '%(A_sky,z) + r'$\Delta z=%s$'%deltaz)
if ell==0: plt.title('Monopole')
if ell==2: plt.title('Quadrupole')
plt.xlim(left=k[0],right=k[-1])
plt.ylim(top=0.5,bottom=-0.5)
plt.figure()
'''

theta_ids = [\
#r'$\overline{T}_{\rm HI}$',\
#r'$b_{\rm HI}$',\
r'$f$',\
r'$D_A$',\
r'$H$',\
r'$A$',\
r'$f_{\rm NL}$'\
]
theta = model.get_param_vals(theta_ids,z,cosmopars)
Fs = []

ells = [0]
F0 = fisher.Matrix_ell(theta_ids,k,Pmod,z,cosmopars,surveypars,V_bin,ells,tracer='X')
Fs.append(F0)

ells = [0,2]
F02 = fisher.Matrix_ell(theta_ids,k,Pmod,z,cosmopars,surveypars,V_bin,ells,tracer='X')
Fs.append(F02)

ells = [0,2,4]
F024 = fisher.Matrix_ell(theta_ids,k,Pmod,z,cosmopars,surveypars,V_bin,ells,tracer='X')
Fs.append(F024)

#Flabels=None
Flabels = [r'$P_0$',r'$P_0 + P_2$',r'$P_0 + P_2 + P_4$']
#Flabels = [r'$P_0 + P_2$',r'$P_0 + P_2 + P_4$']
fisher.CornerPlot(Fs,theta,theta_ids,Flabels)
plt.show()
