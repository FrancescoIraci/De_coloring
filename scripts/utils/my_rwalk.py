"""
This module contains definitions of Metropolis-Hasting all chains
steps with random walks proposals.
"""

from typing import Type

import jax
import jax.numpy as jnp

from jexplore.sampling import EpochMH, StateMH
from jexplore.steps.direct import Direct
from jexplore.tools import distributions as d


class MVRandomWalk(Direct):
    r"""
    Generic multivariate random walk proposal.

    :param dist: distribution class
    :param mask: proposal dimensions mask (default all space)
    :param scale: random walk scaling factor.
    :param \**opts: options to be passed to the distribution creator.
    """

    lower: jax.Array
    """covariance cholesky decomposition matrices `(nchains, dim, dim)`."""

    cd_const: jax.Array
    """ effective scaling factor. This is the `scale` attribute
        divided by the square root of the space dimensionality."""

    scale: float
    """random walk scaling factor."""

    def __init__(
        self,
        dist: Type[d.Distr],
        mask: jax.Array | None = None,
        scale: float = 1.19,
        **opts,
    ):
        super().__init__(mask=mask, dist=dist, **opts)
        self.scale = scale

    def build(self, epoch: EpochMH) -> None:
        """Step epoch initialisation method. This extends
        method :py:attr:`jexplore.steps.step.Step.build` by defining
        the `lower` and `cd_const` attributes.

        `epoch` and `sampling` attributes.
        :param epoch: current epoch.
        """
        super().build(epoch)
        self.lower = jnp.linalg.cholesky(self.masked_covs)
        self.cd_const = self.scale / jnp.sqrt(self.mask.size)

    def proposal(
        self, key: jax.Array, state: StateMH
    ) -> tuple[jax.Array, StateMH, jax.Array]:
        """
        Samples from :py:attr:`jexplore.steps.rwalk.MVRandomWalk.epoch_dist`
        are used to perform random walk steps on all chains.

        :param key: PRNG key used as the random key.
        :param state: a (nwalkers * ntemps, dim) array representing
                      the state point of the MCMC sampler. This corresponds
                      to the :py:attr:`jexplore.sampling.state.State.p`
                      attribute of a :py:attr:`jexplore.sampling.state.State`
                      object.

        :return: the updated PRNG key, the new state point (nwalkers * ntemps, dim)
                 and a (nwalkers *ntemps) array with the log of the ratio between the
                 backward and the forward transition probabilities.
        """

        key, y = self.epoch_dist.sample(key, shape=state.p.shape[:1])
        step = self.cd_const * jnp.einsum("cij,cj->ci", self.lower, y)
        state = state.set_val(step + state.p[:, self.mask], pslc=self.mask)
        return key, state, jnp.zeros((state.p.shape[0]), dtype=jnp.float64)


class GaussianRandomWalk(MVRandomWalk):
    """
    Gaussian random walk proposal.

    :param mask: proposal dimensions mask (default all space)
    """

    def __init__(self, mask: jax.Array | None = None):
        super().__init__(mask=mask, dist=d.Normal)


class StudentTRandomWalk(MVRandomWalk):
    """
    Student-T random walk proposal.

    :param mask: proposal dimensions mask (default all space)
    :param float nu: Student-T nu parameter (default: 5)
    """

    def __init__(self, mask: jax.Array | None = None, nu=5.0):
        super().__init__(mask=mask, dist=d.StudentT, nu=nu)

