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
Survey = 'MK_UHF' # MeerKLASS UHF-band
#Survey = 'SKA'
z,zmin,zmax,R_beam,A_sky,t_tot,N_dish,V_bin = survey.params(Survey)

kmin = np.pi/V_bin**(1/3) ### From Tayura https://arxiv.org/pdf/1101.4723.pdf (after eq8)
kmax = 0.4
kbins = np.arange(kmin,kmax,kmin)
k = (kbins[1:] + kbins[:-1])/2 #centre of k bins

Pmod = cosmo.MatterPk(z,kmin=kmin-kmin/1000)

Omega_HI = model.OmegaHI(z)
Tbar = model.Tbar(z,Omega_HI)
b_HI = model.b_HI(z)
f = cosmo.f(z)
bphiHI = cosmo.b_phi_universality(b_HI)
f_NL = 0

cosmopars = [Omega_HI,b_HI,f,bphiHI,f_NL]
surveypars = [zmin,zmax,R_beam,A_sky,t_tot,N_dish]

### 2D model power check:
kperp = np.linspace(kmin,kmax,400)
kpara = np.linspace(kmin,kmax,400)
kperp,kpara = np.meshgrid(kperp,kpara)
kgrid = np.sqrt(kperp**2 + kpara**2)
mugrid = kpara/kgrid
P_HI2D = model.P_HI_obs(kgrid,mugrid,z,Pmod,cosmopars,surveypars)
plt.imshow(np.log10(P_HI2D),extent=[kmin,kmax,kmin,kmax],origin='lower')
plt.colorbar()
plt.figure()

### Multipole power check:
ells = [0,2,4]
for ell in ells:
    P_HI_obs = model.P_HI_ell_obs(ell,k,z,Pmod,cosmopars,surveypars)
    plt.plot(k,P_HI_obs,label=r'$P^{\rm obs}_{{\rm HI},%s}$'%ell)
plt.axhline(model.P_N(z,zmin,zmax,A_sky,t_tot,N_dish),color='grey',label=r'$P_{\rm N}$')
plt.axhline(Tbar**2*model.P_SN(z),color='black',label=r'$\overline{T}_{\rm HI}^2 P_{\rm SN}$ (no beam)')
plt.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=[0.5,0.98])
plt.loglog()
plt.figure()

theta_ids = [\
#r'$\overline{T}_{\rm HI}$',\
r'$b_{\rm HI}$',\
r'$f$',\
r'$f_{\rm NL}$'\
]

### Plot covariance matrix for k-bins and all multipole permutations:
'''
ells = [0,2,4]
nk = 25
k_covdemo = np.linspace(0.05,0.4,nk) # array of fewer k for covariance demo plot
C = fisher.Cov_ell(ells,k_covdemo,z,Pmod,cosmopars,surveypars)
R = np.zeros((3*nk,3*nk)) # correlation matrix
for i in range(3*nk):
    for j in range(3*nk):
        R[i,j] = C[i,j] / np.sqrt(C[i,i]*C[j,j])
plt.imshow(R)
plt.axvline(nk,color='black')
plt.axvline(2*nk,color='black')
plt.axhline(nk,color='black')
plt.axhline(2*nk,color='black')
plt.colorbar()
plt.figure()
#exit()
'''

#ells = [0]
#F = fisher.Matrix_ell(theta_ids,k,Pmod,z,cosmopars,surveypars,V_bin,ells)
#np.save('data/FishMatexample0',F)
F0 = np.load('data/FishMatexample0.npy')
C0 = np.linalg.inv(F0)

#ells = [2]
#F = fisher.Matrix_ell(theta_ids,k,Pmod,z,cosmopars,surveypars,V_bin,ells)
#np.save('data/FishMatexample2',F)
F2 = np.load('data/FishMatexample2.npy')
C2 = np.linalg.inv(F2)

#ells = [0,2]
#F02 = fisher.Matrix_ell(theta_ids,k,Pmod,z,cosmopars,surveypars,V_bin,ells)
#np.save('data/FishMatexample02',F02)
F02 = np.load('data/FishMatexample02.npy')
C02 = np.linalg.inv(F02)

#ells = [0,2,4]
#F024 = fisher.Matrix_ell(theta_ids,k,Pmod,z,cosmopars,surveypars,V_bin,ells)
#np.save('data/FishMatexample024',F024)
F024 = np.load('data/FishMatexample024.npy')
C024 = np.linalg.inv(F024)

F2d = fisher.Matrix_2D(theta_ids,k,Pmod,z,cosmopars,surveypars,V_bin)

theta = model.get_param_vals(theta_ids,z,cosmopars)

#Fs = [F0,F02,F2d]
Fs = [F0,F02,F024,F2d]

#Flabels = [r'$P_0$',r'$P_0 + P_2$',r'$P_{\rm 2D}$']
#Flabels = [r'$P_0$',r'$P_2$',r'$P_0 + P_2$',r'$P_{\rm 2D}$']
Flabels = [r'$P_0$',r'$P_0 + P_2$',r'$P_0 + P_2 + P_4$',r'$P_{\rm 2D}$']
fisher.CornerPlot(Fs,theta,theta_ids,Flabels)
plt.show()
exit()


Npar = np.shape(C)[0]
R = np.zeros((Npar,Npar)) # correlation matrix
for i in range(Npar):
    for j in range(Npar):
        R[i,j] = C[i,j] / np.sqrt(C[i,i]*C[j,j])
plt.figure()
plt.imshow(R,vmin=-1,vmax=1,cmap='bwr_r')
plt.colorbar()
plt.show()
exit()
