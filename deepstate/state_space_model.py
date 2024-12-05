from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor


class SSM(ABC):
    """Base class for State Space Model.

    The system dynamics is described with the following equations:

    .. math::
       y_t = a^T_t l_{t-1} + b_t + \sigma_t\\varepsilon_t
    .. math::
       l_t = F_t l_{t-1} + g_t\epsilon_t
    .. math::
       l_0 \sim N(\mu_0, diag(\sigma_0^2)), \\varepsilon_t \sim N(0, 1), \epsilon_t \sim N(0, 1),

    where

       :math:`y` - state of the system

       :math:`l` - state of the system in the latent space

       :math:`a` - emission coefficient

       :math:`F` - transition coefficient

       :math:`g` - innovation coefficient

       :math:`\sigma` - noise standard deviation

       :math:`\mu_0` - prior mean

       :math:`\sigma_0` - prior standard deviation
    """

    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension of the latent space."""
        raise NotImplementedError

    @abstractmethod
    def emission_coeff(self, prior_mean) -> Tensor:
        """Emission coefficient matrix."""
        raise NotImplementedError

    @abstractmethod
    def transition_coeff(self) -> Tensor:
        """Transition coefficient matrix."""
        raise NotImplementedError

    @abstractmethod
    def innovation_coeff(self, prior_mean) -> Tensor:
        """Innovation coefficient matrix."""
        raise NotImplementedError


class LevelSSM(SSM):
    """Class for Level State Space Model.

    Note
    ----
    This class requires ``torch`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def latent_dim(self) -> int:
        """Dimension of the latent space.

        Returns
        -------
        :
            Dimension of the latent space.
        """
        return 1

    def emission_coeff(self, prior_mean) -> Tensor:
        """Static emission coefficient matrix."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size, seq_length = prior_mean.shape[0], prior_mean.shape[1]  # Define appropriately
        return torch.ones(batch_size, seq_length, self.latent_dim(), device=device).float()

    def transition_coeff(self) -> Tensor:
        """Static transition coefficient matrix."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.eye(self.latent_dim(), device=device).float()

    def innovation_coeff(self, prior_mean) -> Tensor:
        """Static innovation coefficient matrix."""
        return self.emission_coeff(prior_mean).float()  # Matches emission in this simplified model


class LevelTrendSSM(LevelSSM):
    """Class for Level-Trend State Space Model.

    Note
    ----
    This class requires ``torch`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    def latent_dim(self) -> int:
        """Dimension of the latent space.

        Returns
        -------
        :
            Dimension of the latent space.
        """
        return 2

    def transition_coeff(self) -> Tensor:
        """Transition coefficient matrix.

        Parameters
        ----------
        datetime_index:
            Tensor with the index values.
            Values should be from 0 to seasonal period.

        Returns
        -------
        :
            Transition coefficient matrix.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transition_coeff = torch.eye(self.latent_dim(),device=device)
        transition_coeff[0, 1] = 1
        return transition_coeff.float()
