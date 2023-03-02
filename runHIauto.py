import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
import cosmo
import model
import survey

### Determine survey:
Survey = 'MK_LB' # MeerKLASS L-band
Survey = 'MK_UHF' # MeerKLASS UHF-band
Survey = 'MK_2019' # MeerKAT 2019 pilot survey in L-band
zc,zmin,zmax,R_beam,A_sky,t_tot,N_dish,V_bin = survey.params(Survey)

kmin = np.pi/V_bin**(1/3)
kmax = 0.4
kbins = np.arange(kmin,kmax,kmin)
k = (kbins[1:] + kbins[:-1])/2 #centre of k bins

Pmod = cosmo.GetModelPk(zc,kmin=kmin-kmin/1000)

Omega_HI = model.OmegaHI(zc)
b_HI = model.b_HI(zc)
f = cosmo.f(zc)
bphiHI = cosmo.b_phi_universality(b_HI)
f_NL = 0
cosmopars = [Omega_HI,b_HI,f,bphiHI,f_NL]

surveypars = [zmin,zmax,R_beam,A_sky,t_tot,N_dish]
'''
#surveypars = [zmin,zmax,R_beam,A_sky,None,N_dish] # Use to remove P_N
P_HI,P_HI_obs,k,nmodes = model.P_HI(Pmod,kbins,zc,cosmopars,surveypars)
P_N = model.P_N(zc,zmin,zmax,A_sky,t_tot,N_dish)
plt.plot(k,P_HI)
plt.plot(k,P_HI_obs)
plt.axhline(P_N)
Tbar = model.Tbar(zc,Omega_HI)
plt.axhline(model.P_SN(zc)*Tbar**2,color='black')
plt.loglog()
plt.figure()
#exit()
'''

theta = np.array([['bHI','f'],[b_HI,f]])

import fisher
F = fisher.Matrix(theta,k,Pmod,zc,cosmopars,surveypars,V_bin)
print(F)
plt.imshow(F)
plt.colorbar()
plt.figure()

errx = np.sqrt(linalg.inv(F)[0,0])
erry = np.sqrt(linalg.inv(F)[1,1])

x,y = b_HI,f
w,h,ang = fisher.ellipse_for_fisher_params(0,1,F)
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
