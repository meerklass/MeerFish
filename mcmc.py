import numpy as np
import emcee
import model

def model_power(theta_mc):
    for i in range(len(theta_select)):
        cosmopars_mc[theta_select[i]] = theta_mc[i]
    mod = model.P(k,mu,Pmod,cosmopars_mc,surveypars,tracer)
    return np.swapaxes(mod,0,1) # [mu,k] -> [k,mu]

def lnlike(theta_mc, Pk, Pkerr): #log likelihood
    return -0.5 * model.ChiSquare(Pk,model_power(theta_mc),Pkerr)

def lnprior(theta_mc): #priors
    for i in range(len(theta_select)):
        if theta_select[i]==7: # alpha_perp
            if 0.95 < theta_mc[i] < 1.05: return 0.0
            return -np.inf
        if theta_select[i]==8: # alpha_para
            if 0.95 < theta_mc[i] < 1.05: return 0.0
            return -np.inf
        if theta_select[i]==10: # f_NL
            if -50 < theta_mc[i] < 50: return 0.0
            return -np.inf
    return 0

def lnprob(theta_mc, Pk, Pkerr):
    lp = lnprior(theta_mc)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta_mc,Pk,Pkerr)

def run(k_,mu_,tracer_,Pk,Pkerr,Pmod_,theta,theta_ids,cosmopars_,surveypars_,nwalkers=200,niter=500,ContinueBackend=False,backendfile=None):
    ''' Main run function for MCMC '''
    global k; global mu; global tracer; global Pmod; global cosmopars_mc; global surveypars
    k = k_; mu=mu_; tracer=tracer_; Pmod=Pmod_; cosmopars_mc=cosmopars_; surveypars=surveypars_
    global theta_select
    theta_select = model.get_param_selection(theta_ids)
    ndim = len(theta)
    p0 = np.zeros((nwalkers,ndim))
    for i in range(ndim):
        if theta[i]==0: p0[:,i] = np.random.normal(theta[i],scale=0.1,size=nwalkers)
        else: p0[:,i] = np.random.normal(theta[i],scale=0.1*theta[i],size=nwalkers)

    #backend = emcee.backends.HDFBackend(backendfile)
    #if ContinueBackend==False: backend.reset(nwalkers, ndim)
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, args=(k,Pk,Pkerr))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Pk,Pkerr))

    print("\nRunning production...")
    if ContinueBackend==False: pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    if ContinueBackend==True: #continue from previous chains
        pos, prob, state = sampler.run_mcmc(None, niter, progress=True)
    samples = sampler.chain.reshape((-1, ndim))
    return samples
