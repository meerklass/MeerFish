import numpy as np
import matplotlib.pyplot as plt
import cosmo
import model
import survey

### Determine survey:
Survey = 'MK_LB' # MeerKLASS L-band
Survey = 'MK_UHF' # MeerKLASS UHF-band
A,zmin,zmax,zc,V_sur = survey.params(Survey)

print(V_sur)
print(2400**3)

kmin = np.pi/V_sur**(1/3)
kmax = 0.2
kbins = np.arange(kmin,kmax,kmin)

Pmod = cosmo.GetModelPk(zc,kmin=kmin-kmin/1000)

Omega_HI = model.OmegaHI(zc)
b_HI = model.b_HI(zc)
f = cosmo.f(zc)
bphiHI = cosmo.b_phi_universality(b_HI)
f_NL = 0

theta = [Omega_HI,b_HI,f,bphiHI,f_NL]
Pmod_HI,k,nmodes = model.P_HI(Pmod,kbins,zc,theta)
plt.plot(k,Pmod_HI)

f_NL = 20
theta = [Omega_HI,b_HI,f,bphiHI,f_NL]
Pmod_HI,k,nmodes = model.P_HI(Pmod,kbins,zc,theta)
plt.plot(k,Pmod_HI)

f = 3*f
f_NL = 0
theta = [Omega_HI,b_HI,f,bphiHI,f_NL]
Pmod_HI,k,nmodes = model.P_HI(Pmod,kbins,zc,theta)
plt.plot(k,Pmod_HI)

f = f/3
R_beam = 10
theta = [Omega_HI,b_HI,f,bphiHI,f_NL]
Pmod_HI,k,nmodes = model.P_HI(Pmod,kbins,zc,theta,R_beam=R_beam)
plt.plot(k,Pmod_HI)

plt.loglog()
plt.show()
