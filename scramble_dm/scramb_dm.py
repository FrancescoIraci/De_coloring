import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import pickle, os, json, corner
from enterprise_extensions.deterministic import cw_delay
from fakepta import fake_pta
from fakepta.fake_pta import Pulsar, copy_array, make_fake_array
import fakepta.correlated_noises as cn
from fakepta.correlated_noises import hd, dipole, monopole, curn, anisotropic
import corner, importlib, inspect, time
import jax.numpy as jnp
import jax

# from create_pta import make_pta
# from sampler import get_lnlikelihood, map_params, setup_sampler, save_chains


@jax.jit
def get_max_like_Gmat(toas, residuals, freqs, G, GNG_inv, f_rn, f_dm):

    # N, m = np.shape(psr.Mmat)
    # U, _, _ = np.linalg.svd(psr.Mmat)
    # G = U[:, m:]

    res = G.T @ residuals

    # RN basis
    F_rn_cos = jnp.cos(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn_sin = jnp.sin(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn = jnp.vstack((F_rn_sin, F_rn_cos))

    F_rn = F_rn @ G

    # DM basis
    F_dm_sin = jnp.cos(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm_cos = jnp.sin(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm = jnp.vstack((F_dm_sin, F_dm_cos))

    F_dm = F_dm @ G

    F = jnp.vstack((F_dm, F_rn))

    FNF = F @ GNG_inv @ F.T
    FNr = F @ GNG_inv @ res

    # lnL = -0.5 * (rNr + atFtNFa - 2atFtNr) + k
    # a_ml = (FtNF)^{-1} FtNr
    a_ml = jnp.linalg.solve(FNF, FNr)
    y = res - a_ml @ F
    lnL_ml = -0.5 * y @ GNG_inv @ y

    return a_ml, lnL_ml

@jax.jit
def get_max_like_tmfit(toas, residuals, freqs, Mmat, Ndiag, f_rn, f_dm):

    # N, m = np.shape(psr.Mmat)
    # U, _, _ = np.linalg.svd(psr.Mmat)
    # G = U[:, m:]

    res = residuals

    # RN basis
    F_rn_cos = jnp.cos(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn_sin = jnp.sin(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn = jnp.vstack((F_rn_sin, F_rn_cos))

    # DM basis
    F_dm_sin = jnp.cos(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm_cos = jnp.sin(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm = jnp.vstack((F_dm_sin, F_dm_cos))

    F = jnp.vstack((F_dm, F_rn))
    F = jnp.vstack((F, Mmat.T))

    FNF = F @ jnp.diag(1/Ndiag) @ F.T
    FNr = F @ jnp.diag(1/Ndiag) @ res

    # lnL = -0.5 * (rNr + atFtNFa - 2atFtNr) + k
    # a_ml = (FtNF)^{-1} FtNr
    a_ml = jnp.linalg.solve(FNF, FNr)
    y = res - a_ml @ F
    lnL_ml = -0.5 * y @ jnp.diag(1/Ndiag) @ y

    return a_ml, lnL_ml

@jax.jit
def get_max_like_rndmfit(toas, residuals, freqs, Ndiag, f_rn, f_dm):

    # N, m = np.shape(psr.Mmat)
    # U, _, _ = np.linalg.svd(psr.Mmat)
    # G = U[:, m:]

    res = residuals

    # RN basis
    F_rn_cos = jnp.cos(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn_sin = jnp.sin(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn = jnp.vstack((F_rn_sin, F_rn_cos))

    # DM basis
    F_dm_sin = jnp.cos(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm_cos = jnp.sin(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm = jnp.vstack((F_dm_sin, F_dm_cos))

    F = jnp.vstack((F_dm, F_rn))

    FNF = F @ jnp.diag(1/Ndiag) @ F.T
    FNr = F @ jnp.diag(1/Ndiag) @ res

    # lnL = -0.5 * (rNr + atFtNFa - 2atFtNr) + k
    # a_ml = (FtNF)^{-1} FtNr
    a_ml = jnp.linalg.solve(FNF, FNr)
    y = res - a_ml @ F
    lnL_ml = -0.5 * y @ jnp.diag(1/Ndiag) @ y

    return a_ml, lnL_ml

@jax.jit
def get_tmfit(residuals, Mmat, Ndiag):

    # N, m = np.shape(psr.Mmat)
    # U, _, _ = np.linalg.svd(psr.Mmat)
    # G = U[:, m:]

    res = residuals

    F = Mmat.T

    FNF = F @ jnp.diag(1/Ndiag) @ F.T
    FNr = F @ jnp.diag(1/Ndiag) @ res

    # lnL = -0.5 * (rNr + atFtNFa - 2atFtNr) + k
    # a_ml = (FtNF)^{-1} FtNr
    a_ml = jnp.linalg.solve(FNF, FNr)
    y = res - a_ml @ F
    lnL_ml = -0.5 * y @ jnp.diag(1/Ndiag) @ y

    return a_ml, lnL_ml

def get_N(psr):

    if psr.backends is None:
        toaerrs = np.sqrt(psr.noisedict[psr.name+'_efac']**2 * psr.toaerrs**2 + 10**(2*psr.noisedict[psr.name+'_log10_tnequad']))
    else:
        toaerrs = np.zeros(len(psr.toas))
        for backend in psr.backends:
            mask_backend = psr.backend_flags == backend
            toaerrs[mask_backend] = np.sqrt(psr.noisedict[psr.name+'_'+backend+'_efac']**2 * psr.toaerrs[mask_backend]**2 + 10**(2*psr.noisedict[psr.name+'_'+backend+'_log10_tnequad']))
    white_cov = toaerrs**2
    return white_cov

def get_fit(toas, G, freqs, a_ml, f_rn, f_dm):

    # RN basis
    F_rn_cos = jnp.cos(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn_sin = jnp.sin(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn = jnp.vstack((F_rn_sin, F_rn_cos))

    F_rn = F_rn @ G

    # DM basis
    F_dm_cos = jnp.cos(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm_sin = jnp.sin(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm = jnp.vstack((F_dm_sin, F_dm_cos))

    F_dm = F_dm @ G

    F = jnp.vstack((F_dm, F_rn))
    y = a_ml @ F

    return y

def get_fit_tm(toas, Mmat, freqs, a_ml, f_rn, f_dm):

    # RN basis
    F_rn_cos = jnp.cos(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn_sin = jnp.sin(2*np.pi*f_rn[:, None] * toas[None, :])
    F_rn = jnp.vstack((F_rn_sin, F_rn_cos))

    # DM basis
    F_dm_cos = jnp.cos(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm_sin = jnp.sin(2*np.pi*f_dm[:, None] * toas[None, :]) * (1400 / freqs[None, :]**2)
    F_dm = jnp.vstack((F_dm_sin, F_dm_cos))

    F = jnp.vstack((F_dm, F_rn))
    F = jnp.vstack((F, Mmat.T))

    y = a_ml @ F

    return y
    
def scramb_dm(psr, noisedict, custom_models, f_rn, f_dm, niter=1000, use_Gmat=True):

    toas = jnp.copy(psr.toas)
    residuals = jnp.copy(psr.residuals)
    freqs = jnp.copy(psr.freqs)

    # f_rn = np.arange(1, n_rn+1) / Tspan
    # f_dm = np.arange(1, n_dm+1) / Tspan

    N, m = np.shape(psr.Mmat)
    U, _, _ = np.linalg.svd(psr.Mmat)
    G = U[:, m:]
    G = jnp.array(G)
    m = 0
    G = np.eye(len(toas))

    Mmat = jnp.array(psr.Mmat)

    Ndiag = get_N(psr)
    GNG = G.T @ jnp.diag(Ndiag) @ G
    GNG_inv = jnp.linalg.inv(GNG)

    if use_Gmat:
        a_ml, lnL_ml_0 = get_max_like_Gmat(toas, residuals, freqs, G, GNG_inv, f_rn, f_dm)
    else:
        tm_ml, _ = get_tmfit(residuals, Mmat, np.ones(len(residuals)))
        residuals -= Mmat @ tm_ml
        # a_ml, lnL_ml_0 = get_max_like_tmfit(toas, residuals, freqs, Mmat, Ndiag, f_rn, f_dm)
        a_ml, lnL_ml_0 = get_max_like_rndmfit(toas, residuals, freqs, Ndiag, f_rn, f_dm)

    # vmap_get_max_like = jax.vmap(get_max_like_Gmat, in_axes=(None, None, 0, None, None, None, None))

    freqz = jnp.array([np.random.choice(freqs, size=len(freqs)) for _ in range(niter)])

    # get_max_like_Gmat(toas, residuals, freqs, G, GNG_inv, n_rn, n_dm)
    # a_ml_scramb, lnL_scramble = vmap_get_max_like(toas, residuals, freqz, G, GNG_inv, f_rn, f_dm)

    # scramble
    lnL_scramble = []
    for i in range(niter):

        print('nscramb '+str(i)+'/'+str(niter)+'           ', end='\r')
        freqs = freqz[i]
        if use_Gmat:
            a_ml, lnL_ml = get_max_like_Gmat(toas, residuals, freqs, G, GNG_inv, f_rn, f_dm)
        else:
            a_ml, lnL_ml = get_max_like_rndmfit(toas, residuals, freqs, Ndiag, f_rn, f_dm)

        if not np.isnan(lnL_ml):
            lnL_scramble.append(lnL_ml)

    return lnL_scramble, lnL_ml_0

# plt.hist(np.array(lnL_scramble))
# plt.axvline(lnL_ml_0, color='k', ls='--')
# plt.show()
# # plt.savefig(outdir+'/'+psr.name+'.png', bbox_inches='tight')
# # np.savetxt(outdir+'/'+psr.name+'.txt', np.append(lnL_ml_0, np.array(lnL_scramble)))
# plt.cla()
# plt.clf()