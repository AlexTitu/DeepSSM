import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
import numpy as np
from utils import AudioAnomalyDataset, stratified_split, printDataInfo, collate_fn
from deepstate.deepstate import DeepStateNet
from deepstate.state_space_model import LevelTrendSSM
from training import training_ssm
from test import test_model

torch.manual_seed(19)
random.seed(19)
np.random.seed(19)

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

# Create the test dataset and use the statistics computed from the training dataset
# dev_test_dataset = DCASE2024Dataset('./DCASE2024', ('dev', 'test'), extension='.npy', standardize=False)
# dev_test_dataset.mean = dev_train_dataset.mean
# dev_test_dataset.std = dev_train_dataset.std
# dev_test_dataset.standardize = True
# dev_test_dataset.preGenerateMelSpecs()


machineFolders = os.listdir('D:\Facultate\Master\An I\SEM II\Disertatie\DCASE2024')

# getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device used for training...{}".format(device))

# dev - ReLU / dev_2 Sigmoid+Unet / dev_3 - minimzed Autenc / dev_4 articol / dev_5 extins / dev_6 reparat /
# dev_7 elimin neuronii/ dev_8 5 dar cu relu
if not os.path.exists(f'./dev_ssm/general'):
    os.makedirs(f'./dev_ssm/general')

models_dir = f'./dev_ssm/general'

# initializing hyperparameters
INIT_LR = 0.01  # 0.1
BATCH_SIZE = 30 # 32
EPOCHS = 250

print(f"[INFO] loading data for {models_dir}...")
dev_train_dataset = AudioAnomalyDataset('D:/Facultate/Master/An I/SEM II/Disertatie/DCASE2024', ('dev', 'train'), 8000, 4000, 3,
                                            extension='.wav', standardize=True)

# for 1 type of machine, 1000 train samples
print(f'Number of train Samples: {len(dev_train_dataset)}')
# for each type of machine, we have 100 normal and 100 anomalous sounds
# print(f'Number of test Samples: {len(dev_test_dataset)}')

#input_shape = dev_train_dataset.shapeof(0)
# printDataInfo(dev_train_dataset)

# initializing  dataloaders
print("[INFO] Initializing Dataloaders")
trainDataLoader = DataLoader(dev_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# testDataLoader = DataLoader(dev_test_dataset, batch_size=1)

# calculate steps per epoch for training, validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
# valSteps = len(valDataLoader.dataset) // BATCH_SIZE
# testSteps = len(testDataLoader.dataset) // 1

# print(f"[INFO] Number of steps (train/val/test): {trainSteps}, {valSteps}, {testSteps}")

print(f"[INFO] Number of steps (train/val): {trainSteps}")
"""
for spec, cls in trainDataLoader:
    print(spec.shape)
    print(cls.type)
    break
"""

print("[INFO] initializing DeepSSM...")
# initializing autoencoder

DeepStateModel = DeepStateNet(LevelTrendSSM(), 1, 2, 64, device).to(device)  # 1024/ 4096 / 40
# model = UNet(input_shape, 1024).to(device)
# initializig optimizer
optimizer = torch.optim.Adam(DeepStateModel.parameters(), lr=INIT_LR)

# intializing lr_scheduler
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

H = {
  "total_train_loss": [],
  "time":[]
}

#training_model(models_dir, model, optimizer, scheduler, lossFn, H, EPOCHS, trainDataLoader, trainSteps,
#                 valDataLoader, valSteps, device)

# prev_train_state = torch.load(f"{models_dir}/train_state_dict_CAE_normed.pt")
#  model.load_state_dict(prev_train_state['model_state_dict'])
# optimizer.load_state_dict(prev_train_state['optimizer_state_dict'])
# scheduler.load_state_dict(prev_train_state['lr_scheduler'])
# last_epoch = prev_train_state['epoch']
# H = prev_train_state['train_loss_history']

training_ssm(models_dir, DeepStateModel, optimizer, H, EPOCHS, trainDataLoader, dev_train_dataset, trainSteps, device)

# test_model(models_dir, model, machineFolders, trainDataLoader, trainSteps, lossFn, device)





