import numpy as np
import matplotlib.pyplot as plt
import cosmo
import model
import survey

### Determine survey:
Survey = 'MK_LB' # MeerKLASS L-band
Survey = 'MK_UHF' # MeerKLASS UHF-band
zc,zmin,zmax,R_beam,A_sky,t_tot,N_dish,V_sur = survey.params(Survey)

print(V_sur)
print(2400**3)

kmin = np.pi/V_sur**(1/3)
kmax = 0.4
kbins = np.arange(kmin,kmax,kmin)

Pmod = cosmo.GetModelPk(zc,kmin=kmin-kmin/1000)

Omega_HI = model.OmegaHI(zc)
b_HI = model.b_HI(zc)
f = cosmo.f(zc)
bphiHI = cosmo.b_phi_universality(b_HI)
f_NL = 0
theta = [Omega_HI,b_HI,f,bphiHI,f_NL]

surveypars = [zmin,zmax,R_beam,A_sky,t_tot,N_dish]
#surveypars = [zmin,zmax,R_beam,A_sky,None,N_dish] # Use to remove P_N
Pmod_HI,k,nmodes = model.P_HI(Pmod,kbins,zc,theta,surveypars)

P_N = model.P_N(zc,zmin,zmax,A_sky,t_tot,N_dish)

plt.plot(k,Pmod_HI)
plt.axhline(P_N)
Tbar = model.Tbar(zc,Omega_HI)
plt.axhline(model.P_SN(zc)*Tbar**2,color='black')
plt.loglog()
plt.show()
