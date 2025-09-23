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
#r'$f_{\rm NL}$'\
]
theta = model.get_param_vals(theta_ids,z,cosmopars)

theta_FWHMs = [0.2,0.4,0.8,1.6,2.2]
theta_FWHMs = [0.2,0.4,0.8,1.6]
Fs,Flabels = [],[]
for i in range(len(theta_FWHMs)):
    print(i)
    P_N1_i = model.P_N(z,A_sky1,t_tot,N_dish,theta_FWHM=theta_FWHMs[i],A_pix=0.3)
    surveypars_i = z,V_bin1,V_bin2,V_binX,theta_FWHMs[i],theta_FWHM2,sigma_z1,sigma_z2,P_N1_i,1/nbar
    Fs.append( fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars_i,ells,tracer=tracer) )
    Flabels.append(r'$\theta_{\rm FWHM}{=}%s\,$deg'%theta_FWHMs[i])
fisher.CornerPlot(Fs,theta,theta_ids,Flabels)
plt.figure()

for i in range(len(theta_FWHMs)):
    C = fisher.FisherInverse(Fs[i])
    if i==0: plt.scatter(theta_FWHMs[i],np.sqrt(C[2,2]),color='red',label=r'$\alpha_\perp$',s=12)
    if i==0: plt.scatter(theta_FWHMs[i],np.sqrt(C[3,3]),color='blue',label=r'$\alpha_\parallel$',s=12)
    if i==0: plt.scatter(theta_FWHMs[i],np.sqrt(C[0,0]),color='tab:purple',label=r'$b$',s=6)
    if i==0: plt.scatter(theta_FWHMs[i],np.sqrt(C[1,1]),color='tab:green',label=r'$f$',s=6)
    plt.scatter(theta_FWHMs[i],np.sqrt(C[0,0]),color='tab:purple',s=6)
    plt.scatter(theta_FWHMs[i],np.sqrt(C[1,1]),color='tab:green',s=6)
    plt.scatter(theta_FWHMs[i],np.sqrt(C[2,2]),color='red',s=12)
    plt.scatter(theta_FWHMs[i],np.sqrt(C[3,3]),color='blue',s=12)
plt.legend()
plt.xlabel(r'Beam size ($\theta_{\rm FWHM}$ [deg])')
plt.ylabel(r'Parameter error [$\sigma(\alpha)$]')
plt.title('Impact of increasing beam on parameter inference')
plt.show()
