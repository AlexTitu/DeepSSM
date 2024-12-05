import numpy as np
from typing import Union, List


class LagTransform:
    """Generates series of lags from audio data (numpy arrays) during each batch."""

    def __init__(self, lags: Union[List[int], int]):
        """Create instance of LagTransform.

        Parameters
        ----------
        lags:
            int value or list of values for lags computation; if int, generate range of lags from 1 to given value
        timestamp_length:
            Length of the audio signal (or number of timestamps)
        """
        if isinstance(lags, int):
            if lags < 1:
                raise ValueError("LagTransform works only with positive lags values.")
            self.lags = list(range(1, lags + 1))
        else:
            if any(lag_value < 1 for lag_value in lags):
                raise ValueError("LagTransform works only with positive lags values.")
            self.lags = lags

    def _generate_lagged_data(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate lagged data dynamically for the given audio signal.

        Parameters
        ----------
        audio_data : np.ndarray
            Raw audio data as a numpy array with shape (batch_size, seq_length, features)

        Returns
        -------
        result : np.ndarray
            Transformed audio data with lags.
        """
        # Create lagged versions of the audio data
        lagged_data = []
        for lag in self.lags:
            lagged_audio = np.roll(audio_data, shift=lag, axis=1)  # Shift the audio data by `lag` time steps
            lagged_data.append(lagged_audio)

        # Convert to numpy array
        return np.concatenate(lagged_data, axis=-1)

    def _transform(self, audio_data: np.ndarray) -> np.ndarray:
        """Transform the audio signal with dynamic lag computation.

        Parameters
        ----------
        audio_data : np.ndarray
            Raw audio data as a numpy array, shape (batch_size, seq_length, features).

        Returns
        -------
        result : np.ndarray
            Transformed audio data with lags.
        """
        return self._generate_lagged_data(audio_data)

    def get_lagged_feature_names(self) -> List[str]:
        """Return the list of lagged feature names generated."""
        return [f"lag_{lag}" for lag in self.lags]