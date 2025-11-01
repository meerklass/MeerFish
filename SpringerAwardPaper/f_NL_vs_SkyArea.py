import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/Users/user/Documents/MeerFish')
import cosmo
import model
import fisher
import survey

### For nice plots:
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import mpl_style
plt.style.use(mpl_style.style1)

ells = [0,2,4]
theta_ids = [\
    #r'$\overline{T}_{\rm HI}$',\
    #r'$b_1$',\
    #r'$b_2$',\
    #r'$b^\phi_1$',\
    #r'$b^\phi_2$',\
    r'$f$',\
    #r'$\alpha_\perp$',\
    r'$\alpha_\parallel$',\
    #r'$A_{\rm BAO}$',\
    r'$f_{\rm NL}$'\
    ]


#Survey = 'SKAO'
Survey = 'MeerKLASS'

epsilon = 0.5 # fraction of total observation time kept (i.e. not lost to RFI)
f_tobsloss = 1-epsilon

if Survey=='SKAO': A_skys = np.arange(10000,36000,2000)
if Survey=='MeerKLASS': A_skys = np.arange(1000,15000,1000)

'''
#### Run quick numbers check on f_NL:
if Survey=='MeerKLASS':
    Survey1_arg = 'MK_UHF'
    Survey2_arg = 'Rubin_early'
    zminzmax = None # use default
if Survey=='SKAO':
    Survey1_arg = 'SKA'
    Survey2_arg = 'Rubin'
    zminzmax = [1.5,3]
z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey1_arg,Survey2=Survey2_arg,zminzmax=zminzmax,f_tobsloss=f_tobsloss)
surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
dbeam,dsys,dphotoz = 0,0,0
nuispars = dbeam,dsys,dphotoz

### Cosmological parameters and kbins:
cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology

Tbar1,Tbar2,b1,b2,bphi1,bphi2,f_,a_perp,a_para,A_BAO,f_NL = cosmopars
#b2 = 1 + z
#bphi2 = cosmo.b_phi_universality(b2)
#bphi1 = cosmo.b_phi_HI(z)
#bphi1 = cosmo.b_phi_universality(b1)
cosmopars = np.array([Tbar1,Tbar2,b1,b2,bphi1,bphi2,f_,a_perp,a_para,A_BAO,f_NL])

Pmod = cosmo.MatterPk(z)
k,kbins,kmin,kmax = model.get_kbins(z,zmin1,zmax1,A_sky1)
theta = model.get_param_vals(theta_ids,z,cosmopars)
F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='1')
fisher.CornerPlot(F,theta,theta_ids)
#F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='2')
#fisher.CornerPlot(F,theta,theta_ids)
F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='MT')
fisher.CornerPlot(F,theta,theta_ids)
plt.show()
exit()
'''

#'''
##### Run to save results **[need to run for both Survey choices]** ###########
Fs1,Fs2,FsMT = [],[],[]
for i in range(len(A_skys)):
    print(i,A_skys[i])
    if Survey=='MeerKLASS':
        Survey1_arg = 'MK_UHF'
        Survey2_arg = 'Rubin_early'
        zminzmax = None # use default
    if Survey=='SKAO':
        Survey1_arg = 'SKA'
        Survey2_arg = 'Rubin'
        zminzmax = [1.5,3]
    z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(Survey1=Survey1_arg,Survey2=Survey2_arg,zminzmax=zminzmax,f_tobsloss=f_tobsloss)
    A_sky1 = A_skys[i] # redefine IM sky area
    # Recalcalculate volumes and noise based on new sky area:
    V_bin1 = survey.Vsur(zmin1,zmax1,A_sky1)
    V_bin2 = survey.Vsur(zmin2,zmax2,A_sky2)
    V_binX = survey.Vsur(np.max([zmin1,zmin2]),np.min([zmax1,zmax2]),A_sky1)
    P_N = model.P_N(z,A_sky1,t_tot,N_dish,theta_FWHM=theta_FWHM1)
    surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
    ### Cosmological parameters and kbins:
    cosmopars = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
    Pmod = cosmo.MatterPk(z)
    k,kbins,kmin,kmax = model.get_kbins(z,zmin1,zmax1,A_sky1)

    Fs1.append( fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='1') )
    Fs2.append( fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='2') )
    FsMT.append( fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='MT') )

    Flabels = ['HI','gal','MT']
    Fs = [Fs1[0],Fs2[0],FsMT[0]]
    #theta = model.get_param_vals(theta_ids,z,cosmopars)
    #fisher.CornerPlot(Fs,theta,theta_ids,Flabels=Flabels)
    #plt.show()
    #exit()

np.save('/Users/user/Documents/MeerFish/SpringerAwardPaper/data/Fisher_Asky_%s'%Survey,[Fs1,Fs2,FsMT])
#exit()
#'''
colors = ['tab:blue','tab:orange','tab:green']

theta_ids[0] = r'$f(z)$'
theta_ids[1] = r'$H(z)$'

fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},figsize=(7,9))
def DoPlot(Survey):
    print('---- '+Survey+' ----')
    if Survey=='SKAO': A_skys = np.arange(10000,36000,2000)
    if Survey=='MeerKLASS': A_skys = np.arange(1000,15000,1000)
    if Survey=='SKAO': A_skys = A_skys[2:]
    if Survey=='SKAO': nom_Area = 20000
    if Survey=='MeerKLASS': nom_Area = 4000

    Fs,Fs2,FsMT = np.load('/Users/user/Documents/MeerFish/SpringerAwardPaper/data/Fisher_Asky_%s.npy'%Survey)
    if Survey=='SKAO': Fs = Fs[2:]
    Fs_nom = Fs[A_skys==nom_Area,:,:][0]
    C_nom = fisher.FisherInverse(Fs_nom)
    C = np.zeros_like(Fs)
    for i in range(len(A_skys)):
        C[i] = fisher.FisherInverse(Fs[i])

    if Survey=='MeerKLASS': a0.axvline(nom_Area,color='black',ls='--')
    if Survey=='SKAO': a0.axvline(nom_Area,color='tab:purple',ls='--')
    tabledata = np.zeros((3,np.shape(C)[1]))
    for i in range(np.shape(Fs)[-1]):
        #if i==1: continue # skip H(z)
        if Survey=='MeerKLASS': a0.plot(A_skys,np.sqrt(C_nom[i,i]) / np.sqrt(C[:,i,i]),label=theta_ids[i],color=colors[i])
        if Survey=='SKAO': a0.plot(A_skys,np.sqrt(C_nom[i,i]) / np.sqrt(C[:,i,i]),color=colors[i],lw=1)
        print(theta_ids[i])
        print(np.sqrt(C[:,i,i]))
        print(np.sqrt(C_nom[i,i]) / np.sqrt(C[:,i,i]))
        print('----')

        '''
        if Survey=='MeerKLASS':
            tabledata[0,i] = np.sqrt(C[np.where(A_skys==1000)[0][0],i,i])
            tabledata[1,i] = np.sqrt(C[np.where(A_skys==4000)[0][0],i,i])
            tabledata[2,i] = np.sqrt(C[np.where(A_skys==10000)[0][0],i,i])
        if Survey=='SKAO':
            tabledata[0,i] = np.sqrt(C[np.where(A_skys==10000)[0][0],i,i])
            tabledata[1,i] = np.sqrt(C[np.where(A_skys==20000)[0][0],i,i])
            tabledata[2,i] = np.sqrt(C[np.where(A_skys==30000)[0][0],i,i])
        '''
DoPlot('MeerKLASS')
DoPlot('SKAO')
a0.axvspan(14000, 36000, alpha=0.1, color='purple')
a0.set_xlim(right=36000)
#a0.text(4400,0.43,'Original MeerKLASS proposal',color='black',fontsize=13)
a0.text(4400,0.5,'MeerKLASS white',color='black',fontsize=13)
a0.text(4400,0.43,r'paper $nominal$ area',color='black',fontsize=13)
a0.text(20400,0.43,r'SKAO Red Book $nominal$ area',color='gray',fontsize=13)
a0.text(5500,0.91,'MeerKLASS',fontsize=16,color='black')
a0.text(5500,0.83,'$(2{,}500\,$ hours)',fontsize=14,color='gray')
a0.text(23000,0.91,r'SKAO ($1.5{<}z{<}3$)',fontsize=16,color='tab:purple')
a0.text(23000,0.83,'$(10{,}000\,$hours)',fontsize=14,color='gray')
a0.set_yticks(np.arange(0.4,2.5,0.2))
a0.set_ylim(top=2.3)
a0.legend(framealpha=1,fontsize=18,loc='upper left',borderaxespad=0.1)
a0.set_ylabel(r'Error improvement $[\sigma_{\rm nom}(\theta_i)\,/\,\sigma(\theta_i)]$')
#plt.xlim(A_skys[0],A_skys[-1])
#plt.yticks(np.arange(0.6,1.6,0.1))
a0.grid(axis='y')
a0.set_title(r'Near-term',loc='left',fontsize=18,color='gray')
a0.set_title(r'Next generation',loc='right',fontsize=18,color='tab:purple')

def Plot_f_NL(Survey):
    if Survey=='SKAO': A_skys = np.arange(10000,36000,2000)
    if Survey=='MeerKLASS': A_skys = np.arange(1000,15000,1000)
    if Survey=='SKAO': A_skys = A_skys[2:]
    a1.axvline(4000,color='black',ls='--')
    a1.axvline(20000,color='tab:purple',ls='--')
    Fs,Fs2,FsMT = np.load('/Users/user/Documents/MeerFish/SpringerAwardPaper/data/Fisher_Asky_%s.npy'%Survey)
    if Survey=='SKAO':
        Fs = Fs[2:]
        FsMT = FsMT[2:]
    C = np.zeros_like(Fs)
    for i in range(len(A_skys)):
        C[i] = fisher.FisherInverse(Fs[i])
    if Survey=='MeerKLASS': a1.plot(A_skys,np.sqrt(C[:,-1,-1]),color=colors[-1],label='Single-tracer 21cm')
    if Survey=='SKAO': a1.plot(A_skys,np.sqrt(C[:,-1,-1]),color=colors[-1],lw=1)

    C = np.zeros_like(FsMT)
    for i in range(len(A_skys)):
        C[i] = fisher.FisherInverse(FsMT[i])

    print('-----Multi-tracer (%s):'%Survey)
    for i in range(np.shape(Fs)[-1]):
        print(theta_ids[i])
        print(np.sqrt(C[:,i,i]))

    if Survey=='MeerKLASS': a1.plot(A_skys,np.sqrt(C[:,-1,-1]),color=colors[-1],ls='--',label='Multi-tracer with RO')
    if Survey=='SKAO': a1.plot(A_skys,np.sqrt(C[:,-1,-1]),color=colors[-1],lw=1,ls='--')

Plot_f_NL('MeerKLASS')
Plot_f_NL('SKAO')
a1.legend(fontsize=16,framealpha=1,borderaxespad=0.2)
a1.axvspan(14000, 36000, alpha=0.1, color='purple')
a1.axhline(5.1,lw=2,ls=':',color='tab:red')
a1.text(23500,3.4,'Current state-of-the-art',fontsize=14,color='tab:red')
a1.text(30000,2.2,'(Planck18)',fontsize=14,color='tab:red')
a1.set_xlim(right=36000)
#a1.text(10200,18,'MeerKLASS',fontsize=16,color='black')
#a1.text(27000,1.5,'SKAO',fontsize=16,color='black')
a1.grid(axis='y')
a1.set_yscale('log')
a1.set_xlabel(r'Intensity map sky area [deg$^2$]')
a1.set_ylabel(r'$\sigma({f_{\rm NL}})$')

import matplotlib
import matplotlib.ticker as ticker
a1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
a1.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.subplots_adjust(hspace=0.04)
plt.savefig('/Users/user/Documents/MeerFish/SpringerAwardPaper/plots/params_vs_Area.pdf',bbox_inches='tight')
plt.show()

exit()
### Code to plot a table:
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
tabledata = np.delete(tabledata, (1,2), axis=1)
theta_ids = np.delete(theta_ids, (1,2), axis=0)
ax.table(tabledata, colLabels=theta_ids,rowLabels=['10k','20k','30k'], loc='center')
fig.tight_layout()
plt.show()
