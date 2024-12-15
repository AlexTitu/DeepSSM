from typing import Iterator
import numpy as np
from typing_extensions import TypedDict

import torch
import torch.nn as nn
from deepstate.linear_dynamic_system import LDS
from deepstate.state_space_model import LevelTrendSSM


class DeepStateBatch(TypedDict):
    """Batch specification for DeepStateModel."""

    encoder_real: "torch.Tensor"  # (batch_size, seq_length, input_size)
    decoder_real: "torch.Tensor"  # (batch_size, horizon, input_size)
    encoder_target: "torch.Tensor"  # (batch_size, seq_length, 1)


class DeepStateNet(nn.Module):
    """DeepState network."""

    def __init__(
        self,
        ssm: LevelTrendSSM,
        input_size: int,
        num_layers: int,
        n_samples: int,
        device
    ):
        """Create instance of DeepStateNet.

        Parameters
        ----------
        ssm:
            State Space Model of the system.
        input_size:
            Size of the input feature space: features for RNN part.
        num_layers:
            Number of layers in RNN.
        n_samples:
            Number of samples to use in predictions generation.
        """
        super().__init__()
        self.ssm = ssm
        self.input_size = input_size
        self.num_layers = num_layers
        self.n_samples = n_samples
        self.latent_dim = self.ssm.latent_dim()
        self.device = device

        self.RNN = nn.LSTM(
            num_layers=self.num_layers,
            hidden_size=self.latent_dim,
            input_size=self.input_size,
            batch_first=True,
        )
        self.projectors = nn.ModuleDict(
            dict(
                prior_mean=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                prior_std=nn.Sequential(
                    nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim), nn.Softplus()
                ),
                innovation=nn.Sequential(
                    nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim), nn.Softplus()
                ),
                noise_std=nn.Sequential(nn.Linear(in_features=self.latent_dim, out_features=1), nn.Softplus()),
                offset=nn.Linear(in_features=self.latent_dim, out_features=1),
            )
        )

    def step(self, batch: DeepStateBatch):  # type: ignore
        """Step for loss computation for training or validation.

        Parameters
        ----------
        batch:
            batch of data

        Returns
        -------
        :
            loss, true_target, prediction_target
        """
        encoder_real = batch["encoder_real"].float().to(self.device)  # (batch_size, seq_length, input_size)
        targets = batch["encoder_target"].float().to(self.device)  # (batch_size, seq_length, 1)
        seq_length = targets.shape[1]

        encoder_values = encoder_real

        output, (_, _) = self.RNN(encoder_values)  # (batch_size, seq_length, latent_dim)
        prior_mean = self.projectors["prior_mean"](output[:, 0])
        prior_std = self.projectors["prior_std"](output[:, 0])

        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(self.projectors["prior_mean"](output)),
            transition_coeff=self.ssm.transition_coeff(),
            innovation_coeff=self.ssm.innovation_coeff(self.projectors["prior_mean"](output)) * self.projectors["innovation"](output),
            noise_std=self.projectors["noise_std"](output),
            prior_mean= prior_mean,
            prior_cov=torch.diag_embed(prior_std * prior_std),
            offset=self.projectors["offset"](output),
            seq_length=seq_length,
            latent_dim=self.latent_dim,
        )


        log_likelihood = lds.log_likelihood(targets=targets)
        log_likelihood = torch.mean(torch.sum(log_likelihood, dim=1))

        return -log_likelihood, targets, targets

    def forward(self, x: DeepStateBatch):  # type: ignore
        """Forward pass.

        Parameters
        ----------
        x:
            batch of data

        Returns
        -------
        :
            forecast with shape (batch_size, decoder_length, 1)
        """
        encoder_real = x["encoder_real"].float()  # (batch_size, seq_length, input_size)
        seq_length = encoder_real.shape[1]
        targets = x["encoder_target"][:, :seq_length].float()  # (batch_size, seq_length, 1)
        decoder_real = x["decoder_real"].float()  # (batch_size, horizon, input_size)

        encoder_values = encoder_real
        decoder_values = decoder_real

        output, (h_n, c_n) = self.RNN(encoder_values)  # (batch_size, seq_length, latent_dim)
        prior_mean = self.projectors["prior_mean"](output[:, 0])
        prior_std = self.projectors["prior_std"](output[:, 0])
        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(prior_mean),
            transition_coeff=self.ssm.transition_coeff(),
            innovation_coeff=self.ssm.innovation_coeff(prior_mean) * self.projectors["innovation"](output),
            noise_std=self.projectors["noise_std"](output),
            prior_mean=prior_mean,
            prior_cov=torch.diag_embed(prior_std * prior_std),
            offset=self.projectors["offset"](output),
            seq_length=seq_length,
            latent_dim=self.latent_dim,
        )
        _, prior_mean, prior_cov = lds.kalman_filter(targets=targets)

        output, (_, _) = self.RNN(decoder_values, (h_n, c_n))  # (batch_size, horizon, latent_dim)
        horizon = output.shape[1]
        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(prior_mean),
            transition_coeff=self.ssm.transition_coeff(),
            innovation_coeff=self.ssm.innovation_coeff(prior_mean) * self.projectors["innovation"](output),
            noise_std=self.projectors["noise_std"](output),
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            offset=self.projectors["offset"](output),
            seq_length=horizon,
            latent_dim=self.latent_dim,
        )

        forecast = torch.mean(lds.sample(n_samples=self.n_samples), dim=0)
        return forecast

    def make_samples(self, audio_data: np.ndarray, encoder_length: int, decoder_length: int) -> Iterator[dict]:
        """
        Make samples directly from audio data.

        Parameters
        ----------
        audio_data : np.ndarray
            Raw audio data as a numpy array, assumed to be a single time series.
        encoder_length : int
            Number of time steps for the encoder.
        decoder_length : int
            Number of time steps for the decoder.

        Yields
        ------
        sample : dict
            Dictionary containing encoder and decoder values for each sample.
        """
        total_length = len(audio_data)
        total_sample_length = encoder_length + decoder_length

        start_idx = 0
        while (start_idx + total_sample_length) <= total_length:
            # Define encoder and decoder sequences
            encoder_real = audio_data[start_idx: start_idx + encoder_length].reshape(-1, 1)
            decoder_real = audio_data[start_idx + encoder_length: start_idx + total_sample_length].reshape(-1, 1)

            # Create a sample dictionary
            sample = {
                "encoder_real": torch.tensor(encoder_real, dtype=torch.float32),
                "decoder_real": torch.tensor(decoder_real, dtype=torch.float32),
                "encoder_target": torch.tensor(encoder_real, dtype=torch.float32),
            }

            yield sample
            start_idx += 1  # Move to the next position