import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import random
import librosa
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import time

torch.manual_seed(19)
random.seed(19)
np.random.seed(19)

models_dir = f'./models_dir/'


class DCASE2024Dataset(Dataset):
  def __init__(self, root_dir, datasetType, transform=None, extension='.wav'):
    self.root_dir = root_dir
    self.transform = transform
    self.datasetType = datasetType
    self.datasetSource = []
    self.datasetTarget = []
    self.extension = extension
    self.load_dataset()


  def __len__(self):
    return len(self.datasetSource) + len(self.datasetTarget)

  def load_dataset(self):
    # first, we have the instance_machine folders
    machineData = os.listdir(self.root_dir)
    for machineFolder in machineData:
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
    for audioPath in self.datasetSource + self.datasetTarget:
        mel_spec = self.logmelspec(audioPath)
        # Get the directory and filename without the extension
        directory, filename = os.path.split(os.path.splitext(audioPath)[0])
        # Save the Mel spectrogram as a numpy array in the same directory as the audio file
        np.save(os.path.join(directory, filename + '.npy'), mel_spec)

  def logmelspec(self, path):
    audioFile, sr = librosa.load(path, sr=16000)

    # Ensure audio is 10 seconds long
    if len(audioFile) < sr * 10:
        audioFile = np.pad(audioFile, (0, sr * 10 - len(audioFile)), constant_values=(0, 0))
    else:
        audioFile = audioFile[:sr * 10]

    # Compute the Mel spectrogram with a fixed number of Mel frequency bands and a fixed hop length
    audiomelspec = librosa.feature.melspectrogram(y=audioFile, sr=sr,
                                                  hop_length=626, n_mels=128)
    audiomelspec_db = librosa.power_to_db(audiomelspec)
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
      audioFile = self.logmelspec(audioPath)

    if self.transform:
      audioFile = self.transform(audioFile)

    if audioPath.find('anomaly') != -1:
      return audioFile, 1
    else:
      return audioFile, 0


dev_train_dataset = DCASE2024Dataset('./DCASE2024', ('dev', 'train'), extension='.npy')
#dev_train_dataset.preGenerateMelSpecs()
dev_test_dataset = DCASE2024Dataset('./DCASE2024', ('dev', 'test'), extension='.npy')
#dev_test_dataset.preGenerateMelSpecs()


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
BATCH_SIZE = 5 # 32
EPOCHS = 100

# initializing  dataloaders
print("[INFO] Initializing Dataloaders")
trainDataLoader = DataLoader(dev_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = DataLoader(dev_test_dataset, batch_size=1)

# calculate steps per epoch for training, validation set
trainSteps = len(trainDataLoader.dataset)//BATCH_SIZE
testSteps = len(testDataLoader.dataset)//1

print(f"[INFO] Number of steps (train/test): {trainSteps}, {testSteps}")

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
    self.conv1 = nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
    self.bn3 = nn.BatchNorm2d(128)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(256)
    self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
    self.bn5 = nn.BatchNorm2d(512)

    # variable to store the shape of the output tensor before flattening the ft.
    # it will be used in decoder to reconstruct
    self.shape_before_flatten = (512, 4, 8)

    # compute the flattened size after convolutions
    flattened_size = 4*8*512

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
    self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
    self.bn1 = nn.BatchNorm2d(512)
    self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
    self.bn2 = nn.BatchNorm2d(256)
    self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2,
                                      padding=2, output_padding=1)
    self.bn4 = nn.BatchNorm2d(64)
    self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2,
                                      padding=2, output_padding=1)
    self.bn5 = nn.BatchNorm2d(32)
    # define final convolutional layer to generate output image
    self.conv1 = nn.Conv2d(32, channels, kernel_size=1)
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
model = AutoEncoder(input_shape, 8192).to(device)

# initializig optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

# initializig loss function
lossFn = nn.MSELoss()

H = {
  "total_train_loss": [],
  "time":[]
}
H['time'].append(time.time())
print("[INFO] training the network...")

for e in range(EPOCHS):
  # set the model in training mode
  model.train()
  optimizer.zero_grad()

  train_loss = 0

  for mel_specs, tags in trainDataLoader:
    # sending data to device
    mel_specs = mel_specs.to(device)
    tags = tags.to(device)

    # perform forward pass and calculate loss
    pred_mel_specs = model(mel_specs)
    loss = lossFn(pred_mel_specs, mel_specs)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_loss += loss.cpu().detach().numpy()

  H['total_train_loss'].append(train_loss / trainSteps)
  H['time'].append(time.time())
  if e==19 or e==39 or e==59 or e==79 or e==99:
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': H}, f"{models_dir}/train_state_dict_CAE_{e+1}.pt")

  # print the model training and validation information
  print("[INFO] EPOCH: {}/{} ...".format(e+1, EPOCHS))
  print("Train loss: {:.5f}".format(H['total_train_loss'][e]))
  print("===================================================================")


# finish measuring how long training took
endTime = time.time()
H['time'].append(endTime)
print("[INFO] Total time taken to train the model: {:.2f}s".format(endTime-H['time'][0]))

# switching off autograd for eval
with torch.no_grad():
  # set the model in eval mode
  model.eval()
  losses = []

  for mel_specs, tags in testDataLoader:
    mel_specs = mel_specs.to(device)
    tags = tags.to(device)

    pred_mel_specs = model(mel_specs)
    loss_val = lossFn(pred_mel_specs, mel_specs)
    losses.append(loss_val.cpu().detach().item())

  testLoss = np.array(losses)
  print(f"Total loss mean: {np.mean(testLoss)}")
  print(f"Total loss std: {np.std(testLoss)}")

# plot the training and val losses
previous_state = torch.load(f"{models_dir}/train_state_dict_CAE.pt")
plt.style.use("ggplot")
H = {
    "train_loss":[],
    "time":[]
}

H = previous_state['train_loss_history']

# Plotting loss on train and evaluation
plt.figure("total_loss").clear()
plt.plot(H["train_loss"], label="total_train_loss", linestyle="solid")
plt.title("Evolutia functiei de cost in timpul antrenarii")
plt.xlabel("# Epoca")
plt.ylabel("Cost")
plt.legend(loc="upper right")
plt.savefig(f"{models_dir}/train_val_graph_CAE.png")
