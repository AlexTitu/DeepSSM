
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
import librosa
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from torchvision import transforms
import numpy as np
import time

torch.manual_seed(19)
random.seed(19)
np.random.seed(19)

# Commented out IPython magic to ensure Python compatibility.
# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)

# %cd drive/MyDrive/Disertatie

models_dir = f'./models_dir/'

"""# Information extracted from the task description article

Each recording is a single-channel audio with a duration of 6 to 18 seconds and a sampling rate of 16 kHz. In the Dev they are mainly 10 seconds.

# Rules:

1. Cannot use the test data from the development dataset as support for hyperparameter tuning (different machines)
2. Cannot rely on the id's of the machines found in the dev set | 1 + 2 = First Shot Problem
3. Anomaly Score to calculate = if after certain threshold is an anomaly
4. Source = a lot of data | Target = same machine different conditions, little data - domain adaptation techniques if info is known
5. Info is unknown so must develop domain generalization techniques so it matches test set too

"Detecting anomalies from different domains with a single threshold. These techniques, unlike domain adaptation
techniques, do not require detection of domain shifts or adaptation of the model during the testing phase."

# Evaluation Metric:

**Area Under Reciever Characteristics Curve** to assess the overall detection performance.

**Partial AUC (pAUC)** was utilized to measure performance in a low false-positive rate [0, p], p=0.1.

In domain generalization task, the AUC for each domain and pAUC for each section are calculated.

Notably, all eight teams that outperformed the
baselines in the official scores also surpassed the baselines in the
harmonic mean of the AUCs in the target domain.

https://arxiv.org/pdf/2305.07828.pdf


"""


# defining Early Stopping class
class EarlyStopping():
  def __init__(self, patience=1, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
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
      torch.save(model.state_dict(), f"{models_dir}/CAE_normed_2.pt")
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
    np.save(f'{models_dir}/losses_train_normed_2.npy', np.array(self.best_train))
    np.save(f'{models_dir}/losses_val_normed_2.npy', np.array(self.best_val))

  def loadLossesLocally(self):
    self.best_train = np.load(f'{models_dir}/losses_train_normed_2.npy')
    self.best_val = np.load(f'{models_dir}/losses_val_normed_2.npy')


class DCASE2024Dataset(Dataset):
  def __init__(self, root_dir, datasetType, transform=None, extension='.wav', standardize=False):
    self.root_dir = root_dir
    self.transform = transform
    self.datasetType = datasetType
    self.standardize = standardize
    self.mean = None
    self.std = None
    self.datasetSource = []
    self.datasetTarget = []
    self.labels = []
    self.extension = extension
    self.load_dataset()

    if self.standardize and self.mean is None and self.std is None:
      self.compute_statistics()

  def __len__(self):
    return len(self.datasetSource) + len(self.datasetTarget)

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
                    self.labels.append(index)
                    if recording.find('source') != -1:
                        self.datasetSource.append(os.path.join(machineSounds, recording))
                    elif recording.find('target') != -1:
                        self.datasetTarget.append(os.path.join(machineSounds, recording))
        else:
            continue

  def preGenerateMelSpecs(self):
    for audioPath in self.datasetSource + self.datasetTarget:
        mel_spec = self.logmelspec(audioPath, 'preGenMelSpecs')
        # Get the directory and filename without the extension
        directory, filename = os.path.split(os.path.splitext(audioPath)[0])
        # Save the Mel spectrogram as a numpy array in the same directory as the audio file
        np.save(os.path.join(directory, filename + '.npy'), mel_spec)

  def logmelspec(self, path, origin):
    audioFile, sr = librosa.load(path, sr=16000)

    # Ensure audio is 10 seconds long
    if len(audioFile) < sr * 10:
        audioFile = np.pad(audioFile, (0, sr * 10 - len(audioFile)), constant_values=(0, 0))
    else:
        audioFile = audioFile[:sr * 10]

    # Compute the Mel spectrogram with a fixed number of Mel frequency bands and a fixed hop length
    audiomelspec = librosa.feature.melspectrogram(y=audioFile, sr=sr,
                                                  hop_length=1251, n_mels=128)
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


dev_train_dataset = DCASE2024Dataset('./DCASE2024', ('dev', 'train'), extension='.npy',  standardize=False)
# dev_train_dataset.preGenerateMelSpecs()

# Create the test dataset and use the statistics computed from the training dataset
dev_test_dataset = DCASE2024Dataset('./DCASE2024', ('dev', 'test'), extension='.npy', standardize=False)
# dev_test_dataset.mean = dev_train_dataset.mean
# dev_test_dataset.std = dev_train_dataset.std
# dev_test_dataset.standardize = True
# dev_test_dataset.preGenerateMelSpecs()

# 7 types of machines, each with 1000 train samples
print(f'Number of train Samples: {len(dev_train_dataset)}')
# for each type of machine, we have 100 normal and 100 anomalous sounds
print(f'Number of test Samples: {len(dev_test_dataset)}')


# getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device used for training...{}".format(device))

for index, (spectrogram, audioType) in enumerate(dev_train_dataset):
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
  if index == 0:
    break

for index, (spectrogram, audioType) in enumerate(dev_test_dataset):
  print(spectrogram.shape)
  print(f'Audio type: {audioType}')
  fig, ax = plt.subplots()
  img = librosa.display.specshow(spectrogram[0], x_axis='time',
                         y_axis='mel', sr=16000,
                         fmax=8000, ax=ax)
  fig.colorbar(img, ax=ax, format='%+2.0f dB')
  ax.set(title='Mel-frequency spectrogram')
  plt.show()
  if index == 2:
    break

# initializing hyperparameters
INIT_LR = 0.001 # 0.1
BATCH_SIZE = 64 # 32
EPOCHS = 200


# ensuring even split of classes in train and validation sets
def stratified_split(dataset, validation_size=0.25, random_state=None):
  # Assuming 'dataset.labels' contains your labels
  labels = dataset.labels

  # Create stratified split for test set
  train_index, val_index = train_test_split(
      range(len(labels)),
      test_size=validation_size,
      random_state=random_state,
      stratify=labels)

  return train_index, val_index,


train_index, val_index = stratified_split(dev_train_dataset, random_state=19)

train_dataset = Subset(dev_train_dataset, train_index)
val_dataset = Subset(dev_train_dataset, val_index)

# initializing  dataloaders
print("[INFO] Initializing Dataloaders")
trainDataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(val_dataset,batch_size=BATCH_SIZE)
testDataLoader = DataLoader(dev_test_dataset, batch_size=1)

# calculate steps per epoch for training, validation set
trainSteps = len(trainDataLoader.dataset)//BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE
testSteps = len(testDataLoader.dataset) // 1

print(f"[INFO] Number of steps (train/test): {trainSteps}, {testSteps}")
print(f"[INFO] Number of steps (train/val): {trainSteps}, {valSteps}")

for spec, cls in trainDataLoader:
  print(spec.shape)
  print(cls.type)
  break

"""# Model implementation
Baseline of the DCASE Challenge: https://arxiv.org/pdf/2303.00455.pdf

"""


class Encoder(nn.Module):
  def __init__(self, input_shape, channels, embedding_dim):
    super(Encoder, self).__init__()
    # define convolutional layers
    self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, stride=(2, 2), padding=2)
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=(2, 2), padding=2)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=(2, 2), padding=2)
    self.bn3 = nn.BatchNorm2d(128)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1)
    self.bn4 = nn.BatchNorm2d(256)
    self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=(2, 2), padding=1)
    self.bn5 = nn.BatchNorm2d(512)

    # variable to store the shape of the output tensor before flattening the ft.
    # it will be used in decoder to reconstruct
    self.shape_before_flatten = (512, 4, 4)

    # compute the flattened size after convolutions
    flattened_size = 4*4*512

    self.fc = nn.Linear(flattened_size, embedding_dim)
    self.relu = nn.ReLU()

  def forward(self, x):
    # apply ReLU activations after each convolutional Layer
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.relu(self.bn4(self.conv4(x)))
    x = self.relu(self.bn5(self.conv5(x)))

    # store the shape before flatten
    self.shape_before_flatten = x.shape[1:]

    # flatten the tensor
    x = x.view(x.size(0), -1)

    # apply fully connected layer to generate embeddings
    x = self.fc(x)
    return x


class Decoder(nn.Module):
  def __init__(self, embedding_dim, shape_before_flatten, channels):
    super(Decoder, self).__init__()

    # define fully connected layer to unflatten the embeddings
    self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flatten))
    # store the shape before flatten
    self.reshape_dim = shape_before_flatten

    # define transpose convolutional layers
    self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=(2, 2),
                                      padding=1, output_padding=1)
    self.bn1 = nn.BatchNorm2d(512)
    self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=(2, 2),
                                      padding=1, output_padding=1)
    self.bn2 = nn.BatchNorm2d(256)
    self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=(2, 2),
                                      padding=1, output_padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=5, stride=(2, 2),
                                      padding=2, output_padding=(1, 1))
    self.bn4 = nn.BatchNorm2d(128)
    self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=(2, 2),
                                      padding=2, output_padding=(1, 1))
    self.bn5 = nn.BatchNorm2d(64)
    # define final convolutional layer to generate output image
    self.conv1 = nn.Conv2d(64, channels, kernel_size=1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # apply fully connected layer to unflatten the embeddings
    x = self.fc(x)
    # reshape the tensor to match shape before flatten
    x = x.view(x.size(0), *self.reshape_dim)

    # apply ReLU activations after each transpose convolutional layer
    x = self.relu(self.bn1(self.deconv1(x)))
    x = self.relu(self.bn2(self.deconv2(x)))
    x = self.relu(self.bn3(self.deconv3(x)))
    x = self.relu(self.bn4(self.deconv4(x)))
    x = self.relu(self.bn5(self.deconv5(x)))

    # apply sigmoid activation to the final convolutional layer to generate output image
    x = self.relu(self.conv1(x))

    return x


class AutoEncoder(nn.Module):
  def __init__(self, input_shape, embedding_dim):
    super(AutoEncoder, self).__init__()
    self.Encoder = Encoder(input_shape, input_shape[0], embedding_dim)
    self.Decoder = Decoder(embedding_dim, self.Encoder.shape_before_flatten, input_shape[0])

  def forward(self, x):
    features = self.Encoder(x)
    x = self.Decoder(features)

    return x


print("[INFO] initializing AutoEncoder...")
# initializing autoencoder
# prev_train_state = torch.load()
model = AutoEncoder(input_shape, 4096).to(device)

# initializig optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

# intializing lr_scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# initializig loss function
lossFn = nn.MSELoss()
"""
H = {
  "total_train_loss": [],
  "total_val_loss":[],
  "time":[]
}
H['time'].append(time.time())
print("[INFO] training the network...")
early_stopper = EarlyStopping(patience=10)

for e in range(EPOCHS):
  # set the model in training mode
  model.train()
  optimizer.zero_grad()

  train_loss = 0
  val_loss = 0

  for mel_specs, _ in trainDataLoader:
    # sending data to device
    mel_specs = mel_specs.to(device)
    # tags = tags.to(device)

    # perform forward pass and calculate loss
    pred_mel_specs = model(mel_specs)
    loss = lossFn(pred_mel_specs, mel_specs)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_loss += loss.cpu().detach().numpy()

  with torch.no_grad():
    model.eval()

    for mel_specs, _ in valDataLoader:
      # sending data to device
      mel_specs = mel_specs.to(device)
      # tags = tags.to(device)

      # perform forward pass and calculate loss
      pred_mel_specs = model(mel_specs)
      loss = lossFn(pred_mel_specs, mel_specs)

      val_loss += loss.cpu().detach().numpy()

  H['total_train_loss'].append(train_loss / trainSteps)
  H['total_val_loss'].append(val_loss / valSteps)
  H['time'].append(time.time())

  scheduler.step(H['total_val_loss'][e])

  torch.save({
      'epoch': e,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'lr_scheduler': scheduler.state_dict(),
      'train_loss_history': H}, f"{models_dir}/train_state_dict_CAE_normed_2.pt")

  # print the model training and validation information
  print("[INFO] EPOCH: {}/{} ...".format(e+1, EPOCHS))
  print("Train loss: {:.5f}".format(H['total_train_loss'][e]))
  print("Val loss: {:.5f}".format(H['total_val_loss'][e]))
  # checking if resulting loss in evaluation improved
  if early_stopper.earlyStop((e + 1), H['total_train_loss'][e], H['total_val_loss'][e], model):
    # if not improved - stop the training
    print("[INFO] Early Stopping the train process. Patience exceeded!")
    print("=============================================================")
    break

  print("=============================================================")


# finish measuring how long training took
endTime = time.time()
print("[INFO] Total time taken to train the model: {:.2f}s".format(endTime-H['time'][0]))
print("[INFO] Best loss was found in Epoch {} where the performance was {:.5f}. "
      "Model's parameters saved!".format(early_stopper.getBestEpoch(), early_stopper.getBestValLoss()))

early_stopper.saveLossesLocally()

"""

# plot the training and val losses
previous_state = torch.load(f"./{models_dir}/train_state_dict_CAE_normed_2.pt")
previous_model = torch.load(f"./{models_dir}/CAE_normed_2.pt")
model.load_state_dict(previous_model)

plt.style.use("ggplot")
H = {
    "train_loss":[],
    "time":[]
}

H = previous_state['train_loss_history']

# Plotting loss on train and evaluation
plt.figure("total_loss").clear()
plt.plot(H["total_train_loss"], label="total_train_loss", linestyle="solid")
plt.plot(H["total_val_loss"], label="total_val_loss", linestyle="solid")
plt.title("Evolutia functiei de cost in timpul antrenarii")
plt.xlabel("# Epoca")
plt.ylabel("Cost")
plt.legend(loc="upper right")
plt.savefig(f"{models_dir}/train_val_graph_CAE_normed_2.png")


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
    print(np.sum(np.array(trainLosses))/trainSteps)

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
  'train_loss_history': H}, f"{models_dir}/train_history_values_normed_2.pt")


# switching off autograd for eval
with torch.no_grad():
  # set the model in eval mode
  model.eval()
  totalTestLoss = 0
  pred_tags = []
  true_tags = []

  for mel_specs, tags in testDataLoader:
    mel_specs = mel_specs.to(device)
    tags = tags.to(device)

    pred_mel_specs = model(mel_specs)
    loss_val = lossFn(pred_mel_specs, mel_specs)
    totalTestLoss += loss_val.cpu().detach().item()

    true_tags.append(tags.cpu().detach().item())
    # If the reconstruction error is greater than the threshold, mark it as an anomaly
    pred_tags.append((loss_val.item() > threshold).astype(int))

  print(f"Total loss mean: {totalTestLoss/testSteps}")

  # anomalies is a list of booleans indicating whether each test sample is considered an anomaly
  print(f"Number of anomalies: {np.sum(pred_tags)}")


fpr, tpr, _ = metrics.roc_curve(true_tags, pred_tags)
roc_auc = metrics.auc(fpr, tpr)

# Compute the indices of the FPR values that are less than 0.1
indices = np.where(fpr <= 0.1)[0]

# Compute the pAUC using these indices
pauc = metrics.auc(fpr[indices], tpr[indices])

print(f"pAUC: {pauc}")
print(f"AUC: {roc_auc}")

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

