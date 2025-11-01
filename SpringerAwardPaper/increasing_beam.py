import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'/Users/user/Documents/MeerFish')
import cosmo
import survey

### For nice plots:
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import mpl_style
plt.style.use(mpl_style.style1)

### Survey parameters:
zminzmax = [1,1.2]
#zminzmax = [0.4,0.6]
epsilon = 0.5 # fraction of total observation time kept (i.e. not lost to RFI)
f_tobsloss = 1-epsilon
Survey2 = 'DESI-ELG'
z,zmin1,zmin2,zmax1,zmax2,A_sky1,A_sky2,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,t_tot,N_dish,sigma_z1,sigma_z2,P_N,nbar = survey.params(zminzmax=zminzmax,f_tobsloss=f_tobsloss,Survey2=Survey2)
surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar

nbar /= 20

dbeam,dsys,dphotoz = 0,0,0
nuispars = dbeam,dsys,dphotoz

### Cosmological parameters:
Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL = cosmo.SetCosmology(z=z,return_cosmopars=True) # set initial default cosmology
cosmopars = np.array([Tbar1,Tbar2,b1,b2,bphi1,bphi2,f,a_perp,a_para,A_BAO,f_NL])
Pmod = cosmo.MatterPk(z)

### k-bins:
kmin = 0.02
kmax = 0.3
kbins = np.linspace(kmin,kmax,200)
k = (kbins[1:] + kbins[:-1])/2 #centre of k bins

dampsignal = True
beamnuis = False

ells = [0,2,4]
#ells = [0,2,4,6,8]
tracer  = '1'
#tracer  = 'MT'
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
r'$\delta_{\rm b}$',\
#r'$\delta_{\rm sys}$',\
#r'$\delta_{\rm z}$',\
]
theta_ids
if beamnuis==False: theta_ids = theta_ids[:-1]
theta = model.get_param_vals(theta_ids,z,cosmopars)

### Quick check on constraints:
'''
theta_FWHM1 = 0.8
surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHM1,theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells=ells,tracer=tracer,dampsignal_=dampsignal)

if beamnuis==True: # apply beam prior:
    priors = [None,None,None,None,5]
    fisher.apply_priors(F,priors,params=[b1,f,a_perp,a_para,theta_FWHM1])
fisher.CornerPlot(F,theta,theta_ids)
plt.show()
exit()
F_2 = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='2',dampsignal_=dampsignal)
F_2[-1,-1] += 1.0 / beam_prior_sigma**2
F_2[0,0] += 1.0 / bias_prior_sigma**2
F_MT = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells,tracer='MT',dampsignal_=dampsignal)
F_MT[-1,-1] += 1.0 / beam_prior_sigma**2
F_MT[0,0] += 1.0 / bias_prior_sigma**2
fisher.CornerPlot([F,F_2,F_MT],theta,theta_ids)
plt.show()
exit()
'''

theta_FWHMs = np.linspace(1e-30,1,50)
'''
pars = np.zeros((len(ells),len(theta_FWHMs),len(theta)))
for l in range(len(ells)):
    if l==0: continue # no constraints possible from monopole
    for i in range(len(theta_FWHMs)):
        print(l,i)
        surveypars = z,V_bin1,V_bin2,V_binX,theta_FWHMs[i],theta_FWHM2,sigma_z1,sigma_z2,P_N,1/nbar
        F = fisher.Matrix_ell(theta_ids,k,Pmod,cosmopars,surveypars,nuispars,ells[:(l+1)],tracer=tracer,dampsignal_=dampsignal)

        if beamnuis==True: # apply beam prior:
            priors = [None,None,None,None,1]
            fisher.apply_priors(F,priors,params=[b1,f,a_perp,a_para,theta_FWHMs[i]])

        C = fisher.FisherInverse(F)
        for p in range(len(theta)):
            pars[l,i,p] = 100*np.sqrt(C[p,p])/theta[p]
#np.save('/Users/user/Documents/MeerFish/SpringerAwardPaper/data/inceasing_beam_dampsignal=%s_beamnuis=%s'%(dampsignal,beamnuis),pars)
'''
ls = ['-','--','-']
colors = ['black','tab:blue','tab:orange','tab:green','tab:purple']
theta_ids[0] = r'$b_{\rm HI}$'
'''
##### single plot for quick checks:
for l in range(len(ells)):
    if l==0: continue # no constraints possible from monopole
    for p in range(len(theta)):
        if theta_ids[p]==r'$b_{\rm HI}$': continue # skip b_HI
        if theta_ids[p]==r'$\delta_{\rm b}$': continue # skip beam nuisance params
        if l==2: plt.plot(theta_FWHMs,pars[l,:,p],label=theta_ids[p],ls=ls[l],color=colors[p])
        else: plt.plot(theta_FWHMs,pars[l,:,p],ls=ls[l],color=colors[p])
plt.show()
exit()
'''

plt.figure(figsize=(16,8))
for i in range(3):
    if i==0: dampsignal,beamnuis = False,False
    if i==1: dampsignal,beamnuis = True,False
    if i==2: dampsignal,beamnuis = True,True
    pars = np.load('/Users/user/Documents/MeerFish/SpringerAwardPaper/data/inceasing_beam_dampsignal=%s_beamnuis=%s.npy'%(dampsignal,beamnuis))

    plt.subplot(131+i)
    plt.grid()
    # Start with dummy legend lines:
    plt.plot([-1,-1],[0,0],color='gray',ls='--')
    plt.plot([-1,-1],[0,0],color='gray',ls='-')
    ax = plt.gca()
    lines = ax.get_lines()

    for l in range(len(ells)):
        if l==0: continue # no constraints possible from monopole
        for p in range(len(theta)):
            if theta_ids[p]==r'$b_{\rm HI}$': continue # skip b_HI
            if theta_ids[p]==r'$\delta_{\rm b}$': continue # skip beam nuisance params
            if l==2: plt.plot(theta_FWHMs,pars[l,:,p],label=theta_ids[p],ls=ls[l],color=colors[p])
            else: plt.plot(theta_FWHMs,pars[l,:,p],ls=ls[l],color=colors[p])

    '''
    ### Combine 2 legends on one figure:
    legend1 = plt.legend(framealpha=1,loc='upper left',handlelength=1.2,borderaxespad=0.1)
    legend2 = plt.legend([lines[i] for i in [0,1]], [r'$P_0{+}P_2$',r'$P_0{+}P_2{+}P_4$'],framealpha=1,loc='upper right',handlelength=1.2,borderaxespad=0.1)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    '''
    if i==0: plt.legend(framealpha=1,loc='upper left',handlelength=1.2,borderaxespad=0.1)
    if i==1: plt.legend([lines[i] for i in [0,1]], [r'$P_0{+}P_2$',r'$P_0{+}P_2{+}P_4$'],framealpha=1,loc='upper left',handlelength=1.2,borderaxespad=0.1)

    plt.xlim(0,theta_FWHMs[-1])
    plt.ylim(0,3)
    plt.xlabel(r'Beam size ($\theta_{\rm FWHM}$ [deg])')
    if i==0: plt.ylabel(r'Parameter error % [$\sigma(\theta_i)\,/\,\theta_i$]')
    if i>0: plt.yticks([0,0.5,1,1.5,2,2.5,3],['','','','','','',''])
    if i>0: plt.xticks([0.25,0.50,0.75,1])
    if dampsignal==False: plt.title('Deconvolved beam',fontsize=20)
    if dampsignal==True and beamnuis==False: plt.title('Fixed beam in signal',fontsize=20)
    if dampsignal==True and beamnuis==True : plt.title('Marginalised beam in signal (with prior)',fontsize=20)
plt.subplots_adjust(wspace=0)
#plt.savefig('/Users/user/Documents/MeerFish/SpringerAwardPaper/plots/increasingbeam_dampsignal=%s_beamnuis=%s.pdf'%(dampsignal,beamnuis))
plt.savefig('/Users/user/Documents/MeerFish/SpringerAwardPaper/plots/increasingbeam.pdf')
plt.show()
