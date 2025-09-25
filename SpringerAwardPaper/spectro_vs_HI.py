import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import sys
sys.path.insert(1,'/Users/user/Documents/FullSkyIM')
import init
import cosmo
import model
import fisher
import survey

### DESI tracers:
# - below from DESI DR2 paper (2503.14738) Table 2 and 3:
zbins_LRG = [0.4,0.6,0.8,1.1]
zbins_ELG = [0.8,1.1,1.6]
zbins_QSO = [0.8,2.1]
N_LRG = [1052151,1613562,1802770]
N_ELG = [2737573,3797271]
N_QSO = [1461588]
A_sky_LRG = 10031
A_sky_ELG = 10352
A_sky_QSO = 11181

ell = 0
tracer = '1'
k = np.linspace(0.005,0.1,400)
k_nom = 0.02 # nominal k at which to measure SNR on power spectrum

epsilon = 0.5 # fraction of total observation time kept (i.e. not lost to RFI)
f_tobsloss = 1-epsilon
b_g = 1.5

'''
SNR_SKAO,SNR_MK = [],[]
z_21cm,z_MK = [],[]
### LRG:
SNR_LRG = np.zeros(len(zbins_LRG)-1)
for i in range(len(zbins_LRG)-1):
    # SKAO:
    zminzmax=[zbins_LRG[i],zbins_LRG[i+1]]
    Survey = 'SKA'
    z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey,zminzmax=zminzmax,f_tobsloss=f_tobsloss)
    V_bin2 = survey.Vsur(zbins_LRG[i],zbins_LRG[i+1],A_sky_LRG)
    nbar = N_LRG[i]/V_bin2
    surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
    ### Cosmological parameters and kbins:
    cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
    Pmod = cosmo.MatterPk(z)

    P_HI = model.P_ell(ell,k,Pmod,cosmopars,surveypars,'1')
    P_HI_err = model.P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,'1')
    SNR_SKAO.append( P_HI[k>k_nom][0] / P_HI_err[k>k_nom][0] )
    z_21cm.append(z)

    cosmopars[1] = 1 # neutralise Tbar parameter for galaxy tracer
    cosmopars[3] = b_g
    P_gal = model.P_ell(ell,k,Pmod,cosmopars,surveypars,'2')
    P_gal_err = model.P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,'2')
    SNR_LRG[i] = P_gal[k>k_nom][0] / P_gal_err[k>k_nom][0]

    # MeerKLASS:
    Survey = 'MK_UHF'
    z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey,zminzmax=zminzmax,f_tobsloss=f_tobsloss)
    surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
    ### Cosmological parameters and kbins:
    cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
    Pmod = cosmo.MatterPk(z)
    P_HI = model.P_ell(ell,k,Pmod,cosmopars,surveypars,'1')
    P_HI_err = model.P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,'1')
    SNR_MK.append( P_HI[k>k_nom][0] / P_HI_err[k>k_nom][0] )
    z_MK.append(z)

### ELG:
SNR_ELG = np.zeros(len(zbins_ELG)-1)
for i in range(len(zbins_ELG)-1):
    # SKAO:
    zminzmax=[zbins_ELG[i],zbins_ELG[i+1]]
    Survey = 'SKA'
    z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey,zminzmax=zminzmax,f_tobsloss=f_tobsloss)
    V_bin2 = survey.Vsur(zbins_ELG[i],zbins_ELG[i+1],A_sky_ELG)
    nbar = N_ELG[i]/V_bin2
    surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
    ### Cosmological parameters and kbins:
    cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
    Pmod = cosmo.MatterPk(z)

    P_HI = model.P_ell(ell,k,Pmod,cosmopars,surveypars,'1')
    P_HI_err = model.P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,'1')
    SNR_SKAO.append( P_HI[k>k_nom][0] / P_HI_err[k>k_nom][0] )
    z_21cm.append(z)

    cosmopars[1] = 1 # neutralise Tbar parameter for galaxy tracer
    cosmopars[3] = b_g
    P_gal = model.P_ell(ell,k,Pmod,cosmopars,surveypars,'2')
    P_gal_err = model.P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,'2')
    SNR_ELG[i] = P_gal[k>k_nom][0] / P_gal_err[k>k_nom][0]

    # MeerKLASS:
    Survey = 'MK_UHF'
    if zbins_ELG[i+1]>1.45: zminzmax = [zbins_ELG[i],1.45]
    z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey,zminzmax=zminzmax,f_tobsloss=f_tobsloss)
    surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
    ### Cosmological parameters and kbins:
    cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
    Pmod = cosmo.MatterPk(z)
    P_HI = model.P_ell(ell,k,Pmod,cosmopars,surveypars,'1')
    P_HI_err = model.P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,'1')
    SNR_MK.append( P_HI[k>k_nom][0] / P_HI_err[k>k_nom][0] )
    z_MK.append(z)

### QSO:
SNR_QSO = np.zeros(len(zbins_QSO)-1)
for i in range(len(zbins_QSO)-1):
    # SKAO:
    zminzmax=[zbins_QSO[i],zbins_QSO[i+1]]
    Survey = 'SKA'
    z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey,zminzmax=zminzmax,f_tobsloss=f_tobsloss)
    V_bin2 = survey.Vsur(zbins_QSO[i],zbins_QSO[i+1],A_sky_QSO)
    nbar = N_QSO[i]/V_bin2
    surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
    ### Cosmological parameters and kbins:
    cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
    Pmod = cosmo.MatterPk(z)

    P_HI = model.P_ell(ell,k,Pmod,cosmopars,surveypars,'1')
    P_HI_err = model.P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,'1')
    SNR_SKAO.append( P_HI[k>k_nom][0] / P_HI_err[k>k_nom][0] )
    z_21cm.append(z)

    cosmopars[1] = 1 # neutralise Tbar parameter for galaxy tracer
    cosmopars[3] = b_g
    P_gal = model.P_ell(ell,k,Pmod,cosmopars,surveypars,'2')
    P_gal_err = model.P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,'2')
    SNR_QSO[i] = P_gal[k>k_nom][0] / P_gal_err[k>k_nom][0]

# High-z SKAO bin:
Survey = 'SKA'
zminzmax = [1.6,3]
z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey,zminzmax=zminzmax,f_tobsloss=f_tobsloss)
surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
### Cosmological parameters and kbins:
cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
Pmod = cosmo.MatterPk(z)

P_HI = model.P_ell(ell,k,Pmod,cosmopars,surveypars,'1')
P_HI_err = model.P_ell_err(ell,k,z,Pmod,cosmopars,surveypars,'1')
SNR_SKAO.append( P_HI[k>k_nom][0] / P_HI_err[k>k_nom][0] )
z_21cm.append(z)

np.save('/Users/user/Documents/MeerFish/SpringerAwardPaper/data/spectroz_vs_HI_data',[SNR_SKAO,SNR_MK,SNR_LRG,SNR_ELG,SNR_QSO,z_21cm,z_MK])
'''
SNR_SKAO,SNR_MK,SNR_LRG,SNR_ELG,SNR_QSO,z_21cm,z_MK = np.load('/Users/user/Documents/MeerFish/SpringerAwardPaper/data/spectroz_vs_HI_data.npy',allow_pickle=True)

plt.figure(figsize=(7,4.5))
#plt.figure(figsize=(7,9)) # For Spinger Awards

plt.bar(zbins_LRG[:-1],SNR_LRG,width=np.diff(zbins_LRG),align='edge',linewidth=0,edgecolor='black',alpha=0.5,label='DESI LRG',hatch="X")
plt.bar(zbins_ELG[:-1],SNR_ELG,width=np.diff(zbins_ELG),align='edge',linewidth=0,edgecolor='black',zorder=-10,alpha=0.5,label='DESI ELG')
plt.bar(zbins_QSO[:-1],SNR_QSO,width=np.diff(zbins_QSO),align='edge',linewidth=0,edgecolor='black',alpha=0.5,label='DESI QSO',hatch="+")

legend1 = plt.legend(bbox_to_anchor=(1,0.6),fontsize=16,frameon=True,loc='center right',handlelength=1.3,handletextpad=0.2,borderaxespad=0.2,handleheight=1.2,labelspacing=0.8,framealpha=1)

plt.plot(z_MK,SNR_MK,color='black',ls='--',label='MeerKLASS')
plt.plot(z_21cm,SNR_SKAO,color='gray',ls=':',label='Next gen. (SKAO)',lw=1)
ax = plt.gca()
lines = ax.get_lines()

## MeerKLASS results:
plt.errorbar(z_MK[:3], SNR_MK[:3], xerr=np.diff(zbins_LRG)/2, fmt="o",color='black',lw=2)
#lt.errorbar(z_MK[3:4], SNR_MK[3:4], xerr=np.diff(zbins_ELG)[0]/2, fmt="o",color='black',lw=2)
plt.errorbar(z_MK[-1], SNR_MK[-1], xerr=(1.45-zbins_ELG[-2])/2, fmt="o",color='black',lw=2)
plt.scatter(z_MK[:2],SNR_MK[:2],color='tab:blue',zorder=10)
plt.scatter(z_MK[2],SNR_MK[2],marker=MarkerStyle('o', fillstyle='left'),color='tab:blue',zorder=10)
plt.scatter(z_MK[2],SNR_MK[2],marker=MarkerStyle('o', fillstyle='right'),color='tab:orange',zorder=10)
plt.scatter(z_MK[-1],SNR_MK[-1],color='tab:orange',zorder=10)

## SKAO results:
plt.errorbar(z_21cm[:3], SNR_SKAO[:3], xerr=np.diff(zbins_LRG)/2, fmt="o",color='gray',lw=1)
plt.errorbar(z_21cm[3:5], SNR_SKAO[3:5], xerr=np.diff(zbins_ELG)[-2:]/2, fmt="o",color='gray',lw=1)
plt.errorbar(z_21cm[5], SNR_SKAO[5], xerr=np.diff(zbins_QSO)/2, fmt="o",color='gray',lw=1)
plt.errorbar(z_21cm[6], SNR_SKAO[6], xerr=(3-1.6)/2, fmt="o",color='gray',lw=1)
plt.scatter(z_21cm[:2],SNR_SKAO[:2],color='tab:blue',zorder=10)
plt.scatter(z_21cm[2],SNR_SKAO[2],marker=MarkerStyle('o', fillstyle='left'),color='tab:blue',zorder=10)
plt.scatter(z_21cm[2],SNR_SKAO[2],marker=MarkerStyle('o', fillstyle='right'),color='tab:orange',zorder=10)
plt.scatter(z_21cm[4:5],SNR_SKAO[4:5],color='tab:orange',zorder=10)
plt.scatter(z_21cm[-2],SNR_SKAO[-2],color='tab:green',zorder=10)
plt.scatter(z_21cm[-1],SNR_SKAO[-1],color='tab:red',zorder=10)

plt.xlabel('Redshift',fontsize=20)
plt.ylabel(r'$P\,/\,\sigma(P)$ (at $k\,{=}\,%s\,h\,{\rm Mpc}^{-1}$)'%k_nom,fontsize=20)

plt.xscale('log')
plt.yscale('log')
import matplotlib.ticker as mticker
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

legend2 = plt.legend([lines[i] for i in [0,1]], [r'MeerKLASS ($10{,}000\,{\rm deg}^2$, $2{,}500\,{\rm hrs}$)',r'Next gen. (SKAO) ($20{,}000\,{\rm deg}^2$, $10{,}000\,{\rm hrs}$)'], ncol=1,fontsize=16,loc='center',bbox_to_anchor=(0.5,1.14),frameon=True,borderaxespad=0.2,labelspacing=0.3)
for i,text in enumerate(legend2.get_texts()):
    if i==1: text.set_color("gray")
ax.add_artist(legend1)
ax.add_artist(legend2)

plt.xticks([0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,3],['0.4','','0.6','','0.8','','1','1.5','2','3'])
plt.yticks([2,3,4,5,6,7,8,9,10],['2','3','4','','6','','8','','10'])

plt.subplots_adjust(top=0.8)
#plt.subplots_adjust(top=0.55) # Spinger Awards
#plt.savefig('/Users/user/Documents/Fellowships/ERC/plots/21cm_vs_DESI.pdf',pad_inches=1)
#plt.savefig('/Users/user/Documents/MeerFish/SpringerAwardPaper/plots/21cm_vs_DESI.pdf',bbox_inches='tight') # Spinger Awards
plt.savefig('/Users/user/Documents/MeerFish/SpringerAwardPaper/plots/21cm_vs_DESI.pdf',pad_inches=1) # Spinger Awards
plt.show()
