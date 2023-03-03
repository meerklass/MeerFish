import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
import cosmo
import model
import survey

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

P_HI = model.integratePkmu(model.P_HI,k,z,Pmod,cosmopars,surveypars)
P_HI_obs = model.integratePkmu(model.P_HI_obs,k,z,Pmod,cosmopars,surveypars)
plt.plot(k,P_HI,label=r'$P_{\rm HI}$')
plt.plot(k,P_HI_obs,label=r'$P^{\rm obs}_{\rm N}$')
plt.axhline(model.P_N(z,zmin,zmax,A_sky,t_tot,N_dish),color='grey',label=r'$P_{\rm N}$')
plt.axhline(model.P_SN(z)*Tbar**2,color='black',label=r'$\overline{T}_{\rm HI}^2 P_{\rm SN}$')
plt.legend()
plt.loglog()
plt.close()

#theta = np.array([['r$b_{\rm HI}$',r'$f$',r'$f_{\rm NL}$'],[b_HI,f,f_NL]])
theta = np.array([[r'$b_{\rm HI}$',r'$f$',r'$f_{\rm NL}$',r'$T_{\rm HI}$'],[b_HI,f,f_NL,Tbar]])

theta = np.array([['bHI','f'],[b_HI,f]])

import fisher
F = fisher.Matrix(theta,k,Pmod,z,cosmopars,surveypars,V_bin)
C = np.linalg.inv(F)
Npar = np.shape(C)[0]
R = np.zeros((Npar,Npar)) # correlation matrix
for i in range(Npar):
    for j in range(Npar):
        R[i,j] = C[i,j] / np.sqrt(C[i,i]*C[j,j])

print(R)
plt.figure()
plt.imshow(R,vmin=-1,vmax=1,cmap='bwr_r')
plt.colorbar()

errx = np.sqrt(C[0,0])
erry = np.sqrt(C[1,1])
x,y = b_HI,f
w,h,ang = fisher.ContourEllipse(F,0,1)
from matplotlib.patches import Ellipse
plt.figure()
ax = plt.gca()
ellipse = Ellipse(xy=(x,y), width=w, height=h, angle=ang, edgecolor='r', fc='none', lw=2)
ax.add_patch(ellipse)
ax.autoscale()
plt.axvline(b_HI,color='black',lw=1,ls='--')
plt.axhline(f,color='black',lw=1,ls='--')
plt.xlabel(r'$b_{\rm HI}$')
plt.ylabel(r'$f$')
plt.title(r'$\sigma(b_{\rm HI})/b_{\rm HI}=%s$'%np.round(100*errx/b_HI,3)+'%' + r'     $\sigma(f)/f=%s$'%np.round(100*erry/f,3)+'%')
plt.show()
exit()

### f_NL constraints:
errx = np.sqrt(C[0,0])
erry = np.sqrt(C[2,2])
x,y = b_HI,f_NL
w,h,ang = fisher.ContourEllipse(F,0,2)

from matplotlib.patches import Ellipse
plt.figure()
ax = plt.gca()
ellipse = Ellipse(xy=(x,y), width=w, height=h, angle=ang, edgecolor='r', fc='none', lw=2)
ax.add_patch(ellipse)
ax.autoscale()
plt.axvline(b_HI,color='black',lw=1,ls='--')
plt.axhline(f_NL,color='black',lw=1,ls='--')
plt.xlabel(r'$b_{\rm HI}$')
plt.ylabel(r'$f_{\rm NL}$')
#plt.title(r'$\sigma(b_{\rm HI})/b_{\rm HI}=%s$'%np.round(100*errx/b_HI,3)+'%' + r'     $\sigma(f_{\rm NL})/f_{\rm NL}=%s$'%np.round(100*erry/f,3)+'%')
plt.title(r'$f_{\rm NL}=0\pm%s$'%erry)
plt.show()
exit()


fisher.CornerPlot(F,theta)
exit()
