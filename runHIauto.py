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

kmin = np.pi/V_bin**(1/3)
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

ells = [0,2,4]
for ell in ells:
    #P_HI = model.integratePkmu(model.P_HI,k,z,Pmod,cosmopars,surveypars,ell=ell)
    #plt.plot(k,P_HI,label=r'$P_{{\rm HI},%s}$'%ell)
    P_HI_obs = model.integratePkmu(model.P_HI_obs,k,z,Pmod,cosmopars,surveypars,ell=ell)
    plt.plot(k,P_HI_obs,label=r'$P^{\rm obs}_{{\rm HI},%s}$'%ell)
plt.axhline(model.P_N(z,zmin,zmax,A_sky,t_tot,N_dish),color='grey',label=r'$P_{\rm N}$')
plt.axhline(Tbar**2*model.P_SN(z),color='black',label=r'$\overline{T}_{\rm HI}^2 P_{\rm SN}$')
plt.legend()
plt.loglog()
plt.close()
#plt.show()
#exit()

theta = np.array([['bHI','f'],[b_HI,f]])
theta = np.array([['bHI','Tbar'],[b_HI,Tbar]])
theta = np.array([['bHI','f','Tbar'],[b_HI,f,Tbar]])
theta = np.array([['bHI','f','fNL'],[b_HI,f,f_NL]])
#theta = np.array([['bHI','f','fNL','Tbar'],[b_HI,f,f_NL,Tbar]])

F = fisher.Matrix(theta,k,Pmod,z,cosmopars,surveypars,V_bin)

C = np.linalg.inv(F)
Npar = np.shape(C)[0]
R = np.zeros((Npar,Npar)) # correlation matrix
for i in range(Npar):
    for j in range(Npar):
        R[i,j] = C[i,j] / np.sqrt(C[i,i]*C[j,j])
plt.figure()
plt.imshow(R,vmin=-1,vmax=1,cmap='bwr_r')
plt.colorbar()

theta_labels = fisher.labels(theta)

ps = [b_HI,f,f_NL]
#ps = [theta[1,0],theta[1,1],theta[1,2]]

fisher.CornerPlot(F,ps,theta_labels)
exit()
