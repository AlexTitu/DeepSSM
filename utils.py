from torch.utils.data import Dataset
import torch
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from deepstate.lags import LagTransform
from typing import List, Tuple, Dict, Union


class AudioAnomalyDataset(Dataset):
    """Custom Dataset for loading and processing audio data for anomaly detection."""

    def __init__(
            self,
            dataset: Tuple[any, any],
            encoder_length: int,
            decoder_length: int,
            num_fragments_per_file: int,
            overlap_factor:int = 0.5,
            sample_rate: int = 16000,
            extension: str = '.wav',
            standardize: bool = False,
            scaling_range: Tuple[float, float] = (0, 1)
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        root_dir : str
            Root directory containing audio files.
        dataset : Tuple[str, str]
            Type of dataset, e.g., ('train', 'normal').
        encoder_length : int
            Number of time steps for the encoder.
        decoder_length : int
            Number of time steps for the decoder.
        num_fragments_per_file:
            Number of splits done in only one audio file
        overlap_factor : float, optional
            Overlap ratio between consecutive fragments (0.0 to 1.0), by default 0.5.
        sample_rate : int, optional
            Sampling rate for audio files, by default 16000.
        extension : str, optional
            File extension to consider, by default '.wav'.
        standardize : bool, optional
            Whether to apply min-max scaling, by default False.
        scaling_range : Tuple[float, float], optional
            Range for scaling, by default (0, 1).
        """
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.num_fragments_per_file = num_fragments_per_file
        self.overlap = overlap_factor
        self.sample_rate = sample_rate
        self.extension = extension
        self.standardize = standardize
        self.scaling_range = scaling_range
        # self.machineType = 'bearing'
        self.dataset_source = dataset[0]
        self.dataset_target = dataset[1]
        self.start_indices = []

    def get_sample_fragments(self):
        return self.start_indices

    def set_sample_fragments(self, start_indices):
        self.start_indices = start_indices

    def sample_fragments(self, file_length):
        """
        Sample fragments from an audio file using a hybrid approach.

        Parameters
        ----------
        file_length : int
            Total length of the audio file (in samples).
        Returns
        -------
        List[int]
            List of start indices for the fragments.
        """
        # Compute the step size based on overlap
        fragment_length = self.encoder_length + self.decoder_length
        step = int(fragment_length * (1 - self.overlap))

        # Ensure valid overlap
        if step <= 0:
            raise ValueError("Overlap too large. Choose a smaller value.")

        # Determine the range for the first start index
        max_start_index = max(0, file_length - (self.num_fragments_per_file * step + fragment_length))
        if max_start_index <= 0:
            raise ValueError("Audio file is too short for the given fragment length and overlap.")

        # Randomly pick the first start index
        first_start_index = torch.randint(0, max_start_index + 1, (1,)).item()

        # Generate sequential fragments
        start_indices = []
        current_index = first_start_index

        for _ in range(self.num_fragments_per_file):
            # Ensure the fragment fits within the file
            if current_index + fragment_length > file_length:
                break
            start_indices.append(current_index)
            current_index += step

        self.start_indices = start_indices

    def __len__(self) -> int:
        """Return the total number of fragments."""
        return (len(self.dataset_source)+len(self.dataset_target)) * self.num_fragments_per_file

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """Retrieve a fragment and its label."""
        file_idx = idx // self.num_fragments_per_file
        fragment_idx = idx % self.num_fragments_per_file

        if file_idx >= len(self.dataset_source):
            file_idx = len(self.dataset_source) - file_idx
            path = self.dataset_target[file_idx]
        else:
            path = self.dataset_source[file_idx]

        if self.extension == '.npy':
            # Directly load precomputed spectrogram
            spectrogram = np.load(path)  # Shape: (1, 128, 128)
            start = self.start_indices[fragment_idx]
            fragment = spectrogram[:, :,start:start + self.encoder_length + self.decoder_length]
            # Check if padding is required
            if fragment.shape[2] < self.encoder_length + self.decoder_length:
                fragment = np.pad(fragment, ((0, 0), (0, 0), (0, self.encoder_length + self.decoder_length - fragment.shape[2])), mode='constant')

            # Remove the channel dimension and reshape for LSTM
            fragment = fragment.squeeze(0)  # Shape: (128, 128)
            fragment = fragment.T  # Transpose to (time, features), i.e., (128, 128)
            encoder_real = fragment[:self.encoder_length]  # (n_mels, encoder_length)
            decoder_real = fragment[self.encoder_length:self.encoder_length + self.decoder_length]  # (n_mels, decoder_length)

        else:  # Raw audio
            signal, _ = librosa.load(path, sr=self.sample_rate)
            start = self.start_indices[fragment_idx]
            fragment = signal[start:start + self.encoder_length + self.decoder_length]
            # Zero-pad if the fragment is too short
            if len(fragment) < self.encoder_length + self.decoder_length:
                fragment = np.pad(fragment, (0, self.encoder_length + self.decoder_length - len(fragment)))

            # Standardize if required
            if self.standardize:
                min_val, max_val = fragment.min(), fragment.max()
                fragment = (fragment - min_val) / (max_val - min_val)
                fragment = fragment * (self.scaling_range[1] - self.scaling_range[0]) + self.scaling_range[0]

            encoder_real = fragment[:self.encoder_length].reshape(-1, 1)  # (encoder_length, 1)
            decoder_real = fragment[self.encoder_length:].reshape(-1, 1)  # (decoder_length, 1)

        # Convert to PyTorch tensors
        encoder_real_tensor = torch.tensor(encoder_real, dtype=torch.float32)
        decoder_real_tensor = torch.tensor(decoder_real, dtype=torch.float32)
        encoder_target_tensor = encoder_real_tensor.clone()  # Encoder target is same as encoder input

        # Construct sample dictionary
        sample = {
            "encoder_real": encoder_real_tensor,  # Shape depends on input type
            "decoder_real": decoder_real_tensor,
            "encoder_target": encoder_target_tensor,  # Target aligns with encoder input
        }

        # Determine label based on file type
        label = 1 if 'anomaly' in path else 0

        return sample, label


def collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], int]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Custom collate function to combine multiple samples into a batch."""
    encoder_real = torch.stack([item[0]["encoder_real"] for item in batch])  # (batch_size, encoder_length, 1)
    decoder_real = torch.stack([item[0]["decoder_real"] for item in batch])  # (batch_size, decoder_length, 1)
    encoder_target = torch.stack([item[0]["encoder_target"] for item in batch])  # (batch_size, encoder_length, 1)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)  # (batch_size,)

    batch_dict = {
        "encoder_real": encoder_real,
        "decoder_real": decoder_real,
        "encoder_target": encoder_target,
    }

    return batch_dict, labels


# -------------------------------------------------
# -- Dataloader concentrated on the whole dataset --
# --------------------------------------------------
class DCASE2024Dataset(Dataset):
  def __init__(self, root_dir, dataset, transform=None, extension='.wav', standardize=False):
    self.root_dir = root_dir
    self.transform = transform
    self.datasetType = dataset
    self.standardize = standardize
    self.mean = None
    self.std = None
    self.datasetSource = []
    self.datasetTarget = []
    self.extension = extension
    self.load_dataset()

    if self.standardize and self.mean is None and self.std is None:
      self.compute_statistics()

  def __len__(self):
    return len(self.datasetSource) + len(self.datasetTarget)

  def sourceLen(self):
    return len(self.datasetSource)

  def targetLen(self):
    return len(self.datasetTarget)

  def compute_statistics(self):
    all_specs = []
    for audioPath in self.datasetSource + self.datasetTarget:
      if self.extension == '.npy':
        audioFile = np.load(audioPath)
      else:
        audioFile = self.logmelspec(audioPath, 'statistics')
      all_specs.append(audioFile)

    all_specs = np.concatenate(all_specs, axis=0)
    self.mean = np.mean(all_specs, axis=0)
    self.std = np.std(all_specs, axis=0)

  def load_dataset(self):
    # first, we have the instance_machine folders
    machineData = os.listdir(self.root_dir)
    for index, machineFolder in enumerate(machineData):
        folderType, machineType = machineFolder.split("_")
        if folderType == self.datasetType[0]:
            machineSounds = os.path.join(self.root_dir, machineFolder, machineType, self.datasetType[1])
            for recording in os.listdir(machineSounds):
                # Check if the file is a .npy file
                if recording.endswith(self.extension):
                    if recording.find('source') != -1:
                        self.datasetSource.append(os.path.join(machineSounds, recording))
                    elif recording.find('target') != -1:
                        self.datasetTarget.append(os.path.join(machineSounds, recording))
        else:
            continue

  def preGenerateMelSpecs(self):
    if self.extension == '.wav':
      for audioPath in self.datasetSource + self.datasetTarget:
          if not os.path.exists(os.path.splitext(audioPath)[0]+'.npy'):
            mel_spec = self.logmelspec(audioPath, 'preGenMelSpecs')
            # Get the directory and filename without the extension
            directory, filename = os.path.split(os.path.splitext(audioPath)[0])
            # Save the Mel spectrogram as a numpy array in the same directory as the audio file
            np.save(os.path.join(directory, filename + '.npy'), mel_spec)
          else:
            continue
    else:
      print(f'Pre-Generation of Mel Specs is not supported!')
      return

    return

  def shapeof(self, index):
      sample, _ = self.__getitem__(index)
      return np.shape(sample)

  def logmelspec(self, path, origin):
    audioFile, sr = librosa.load(path, sr=16000)

    # Ensure audio is 10 seconds long
    if len(audioFile) < sr * 10:
        audioFile = np.pad(audioFile, (0, sr * 10 - len(audioFile)), constant_values=(0, 0))
    else:
        audioFile = audioFile[:sr * 10]

    # Compute the Mel spectrogram with a fixed number of Mel frequency bands and a fixed hop length
    audiomelspec = librosa.feature.melspectrogram(y=audioFile, sr=sr, n_fft=1024, hop_length=512, n_mels=128) # 1251 / 5049
    audiomelspec_db = librosa.power_to_db(audiomelspec)

    if self.standardize and origin != 'statistics':
      audiomelspec_db = (audiomelspec_db - self.mean) / self.std

    audiomelspec_db = np.array([audiomelspec_db])

    return audiomelspec_db

  def __getitem__(self, index):
    if index >= len(self.datasetSource):
      index = len(self.datasetSource)-index
      audioPath = self.datasetTarget[index]
    else:
      audioPath = self.datasetSource[index]

    if self.extension == '.npy':
      audioFile = np.load(audioPath)
    else:
      audioFile = self.logmelspec(audioPath, 'getitem')

    if self.transform:
      audioFile = self.transform(audioFile)

    if audioPath.find('anomaly') != -1:
      return audioFile, 1
    else:
      return audioFile, 0


# -------------------------------------------------
# -- Dataloader concentrated on the machine type --
# -------------------------------------------------
class DCASE2024MachineDataset(Dataset):
  def __init__(self, root_dir, dataset, machine, transform=None, extension='.wav', standardize=None, isTesting=False):
    self.root_dir = root_dir
    self.transform = transform
    self.datasetType = dataset
    self.machineType = machine.split("_")[1]
    self.isTesting = isTesting

    valid = {'statistical', 'min-max', None}
    if standardize not in valid:
      raise ValueError("Error: standardize must be one of %r." % valid)
    else:
      self.standardize = standardize

    self.mean = None
    self.std = None
    self.datasetSource = []
    self.datasetTarget = []
    self.extension = extension
    self.load_dataset()

    if self.standardize == 'statistical' and self.mean is None and self.std is None:
      self.compute_statistics()

  def __len__(self):
    return len(self.datasetSource) + len(self.datasetTarget)

  def sourceLen(self):
    return len(self.datasetSource)

  def targetLen(self):
    return len(self.datasetTarget)

  def compute_statistics(self):
    all_specs = []
    for audioPath in self.datasetSource + self.datasetTarget:
      if self.extension == '.npy':
        audioFile = np.load(audioPath)
      else:
        audioFile = self.logmelspec(audioPath, 'statistics')
      all_specs.append(audioFile)

    all_specs = np.concatenate(all_specs, axis=0)
    self.mean = np.mean(all_specs, axis=0)
    self.std = np.std(all_specs, axis=0)

  def load_dataset(self):
    # first, we have the instance_machine folders
    machineData = os.listdir(self.root_dir)
    for index, machineFolder in enumerate(machineData):
        folderType, machineType = machineFolder.split("_")
        if folderType == self.datasetType[0] and machineType == self.machineType:
            machineSounds = os.path.join(self.root_dir, machineFolder, machineType, self.datasetType[1])
            for recording in os.listdir(machineSounds):
                # Check if the file is a .npy file
                if recording.endswith(self.extension):
                    if recording.find('source') != -1:
                        self.datasetSource.append(os.path.join(machineSounds, recording))
                    elif recording.find('target') != -1:
                        self.datasetTarget.append(os.path.join(machineSounds, recording))
        else:
            continue

  def preGenerateMelSpecs(self, override=False):
    if self.extension == '.wav':
      for audioPath in self.datasetSource + self.datasetTarget:
          if not os.path.exists(os.path.splitext(audioPath)[0]+'.npy') or override:
            mel_spec = self.logmelspec(audioPath, 'preGenMelSpecs')
            # Get the directory and filename without the extension
            directory, filename = os.path.split(os.path.splitext(audioPath)[0])
            # Save the Mel spectrogram as a numpy array in the same directory as the audio file
            np.save(os.path.join(directory, filename + '.npy'), mel_spec)
          else:
            continue
    else:
      print(f'Pre-Generation of Mel Specs is not supported!')
      return

    return

  def shapeof(self, index):
      sample, _ = self.__getitem__(index)
      return np.shape(sample)

  def logmelspec(self, path, origin):
    audioFile, sr = librosa.load(path, sr=16000)

    # Ensure audio is 10 seconds long
    if len(audioFile) < sr * 10:
        audioFile = np.pad(audioFile, (0, sr * 10 - len(audioFile)), constant_values=(0, 0))
    else:
        audioFile = audioFile[:sr * 10]

    # Compute the Mel spectrogram with a fixed number of Mel frequency bands and a fixed hop length
    audiomelspec = librosa.feature.melspectrogram(y=audioFile, sr=sr, n_fft=1024,
                                                  hop_length=621, n_mels=128) # 1251 / 5049
    audiomelspec_db = librosa.power_to_db(audiomelspec)

    if self.standardize == 'statistical' and origin != 'statistics':
        audiomelspec_db = (audiomelspec_db - self.mean) / self.std

    elif self.standardize == 'min-max':
        audiomelspec_db = ((audiomelspec_db - np.min(audiomelspec_db)) /
                           (np.max(audiomelspec_db) - np.min(audiomelspec_db)))

    audiomelspec_db = np.array([audiomelspec_db])

    return audiomelspec_db

  def __getitem__(self, index):
    if index >= len(self.datasetSource):
      origin = 'target'
      index = len(self.datasetSource)-index
      audioPath = self.datasetTarget[index]
    else:
      origin = 'source'
      audioPath = self.datasetSource[index]

    if self.extension == '.npy':
      audioFile = np.load(audioPath)
    else:
      audioFile = self.logmelspec(audioPath, 'getitem')

    if self.transform:
      audioFile = self.transform(audioFile)

    if not self.isTesting:
        if audioPath.find('anomaly') != -1:
          return audioFile, 1
        else:
          return audioFile, 0
    else:
        if audioPath.find('anomaly') != -1:
            return audioFile, 1, origin
        else:
            return audioFile, 0, origin


# -----------------------------------
# -- Defining Early Stopping Class --
# -----------------------------------
class EarlyStopping():
  def __init__(self, patience=1, min_delta=0, models_dir='./models_dir'):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.models_dir = models_dir
    self.min_validation_loss = np.inf
    self.best_epoch = 0
    self.best_train = None
    self.best_val = None

  def earlyStop(self, epoch, trainLoss, valLoss, model):
    if valLoss <= (self.min_validation_loss + self.min_delta):
      print("[INFO] In EPOCH {} the loss value improved from {:.5f} to {:.5f}".format(epoch, self.min_validation_loss, valLoss))
      self.setMinValLoss(valLoss)
      self.setCounter(0)
      self.setBestEpoch(epoch)
      torch.save(model.state_dict(), f"{self.models_dir}/CAE_normed.pt")
      self.setBestLosses(trainLoss, valLoss)

    elif valLoss > (self.min_validation_loss + self.min_delta):
      self.setCounter(self.counter + 1)
      print("[INFO] In EPOCH {} the loss value did not improve from {:.5f}. This is the {} EPOCH in a row.".format(epoch, self.min_validation_loss, self.counter))
      if self.counter >= self.patience:
        return True
    return False

  def setCounter(self, counter_state):
    self.counter = counter_state

  def setMinValLoss(self, ValLoss):
    self.min_validation_loss = ValLoss

  def setBestLosses(self, trainLoss, valLoss):
    self.best_train = trainLoss
    self.best_val = valLoss

  def setBestEpoch(self, bestEpoch):
    self.best_epoch = bestEpoch

  def getBestTrainLoss(self):
    return self.best_train

  def getBestValLoss(self):
    return self.best_val

  def getBestEpoch(self):
    return self.best_epoch

  def saveLossesLocally(self):
    np.save(f'{self.models_dir}/losses_train_normed.npy', np.array(self.best_train))
    np.save(f'{self.models_dir}/losses_val_normed.npy', np.array(self.best_val))

  def loadLossesLocally(self):
    self.best_train = np.load(f'{self.models_dir}/losses_train_normed.npy')
    self.best_val = np.load(f'{self.models_dir}/losses_val_normed.npy')


# -------------------------------------
# -- Printing and plotting functions --
# -------------------------------------
def plotSpectrogram(spectrogram, audioType):
    input_shape = spectrogram.shape
    print(spectrogram.shape)
    print(f'Audio type: {audioType}')
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram[0], x_axis='time',
                           y_axis='mel', sr=16000,
                           fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def load_dataset(root_dir: str, dataset_type: Tuple[str, str], extension: str):
    """Load dataset paths based on dataset type."""
    dataset_source = []
    dataset_target = []
    machine_data = os.listdir(root_dir)
    for index, machine_folder in enumerate(machine_data):
        folder_type, machine_type = machine_folder.split("_")
        if folder_type == dataset_type[0]:  # and machine_type == self.machineType:
            machine_sounds = os.path.join(root_dir, machine_folder, machine_type, dataset_type[1])
            for recording in os.listdir(machine_sounds):
                if recording.endswith(extension):
                    if 'source' in recording:
                        dataset_source.append(os.path.join(machine_sounds, recording))
                    elif 'target' in recording:
                        dataset_target.append(os.path.join(machine_sounds, recording))

    return dataset_source, dataset_target


# ensuring even split of classes in train and validation sets
def stratified_split(dataset_source: list, dataset_target: list, validation_size: float = 0.25, random_state: int = 42):
    """
        Stratify split only the file paths for source and target datasets.
        """
    # Split source file paths
    source_train, source_val = train_test_split(
        dataset_source,
        test_size=validation_size,
        random_state=random_state
    )

    # Split target file paths
    target_train, target_val = train_test_split(
        dataset_target,
        test_size=validation_size,
        random_state=random_state
    )

    return source_train, source_val, target_train, target_val


def printDataInfo(dataset):
    for spectrogram, tag in dataset:
        print(np.shape(spectrogram))
        print(np.min(spectrogram))
        print(np.max(spectrogram))
        print(tag)
        break


"""
  # switching off autograd for eval
        with torch.no_grad():
          # set the model in eval mode
          model.eval()
          trainLosses = []

          for mel_specs, _ in trainDataLoader:
            mel_specs = mel_specs.to(device)

            pred_mel_specs = model(mel_specs)
            loss_val = lossFn(pred_mel_specs, mel_specs)
            trainLosses.append(loss_val.cpu().detach().item())

        # Assume trainLosses is a list of the reconstruction errors on your training data
        train_losses = np.array(trainLosses)
        mean = np.mean(train_losses)
        std_dev = np.std(train_losses)
        print(f"Total loss mean: {mean}")
        print(f"Total loss std: {std_dev}")

        # Set threshold as mean plus 3 standard deviations
        threshold = mean + 3 * std_dev

        H['threshold'] = threshold

        torch.save({
          'train_loss_history': H}, f"{models_dir}/train_history_values_normed.pt")

"""
