import torch.nn as nn
import numpy as np
import torch

"""# Model implementation
Baseline of the DCASE Challenge: https://arxiv.org/pdf/2303.00455.pdf

"""


class Encoder(nn.Module):
  def __init__(self, input_shape, channels, embedding_dim):
    super(Encoder, self).__init__()
    # define convolutional layers
    self.conv1 = nn.Conv2d(channels, 8, kernel_size=5, stride=(2, 2), padding=2) # (2,1)
    self.bn1 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=(2, 2), padding=2)  # (2,1)
    self.bn2 = nn.BatchNorm2d(16)
    self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=(2, 2), padding=2) #128 (2,1)
    self.bn3 = nn.BatchNorm2d(32)
    self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=(2, 2), padding=2) # 3 pad 1
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1)
    self.bn5 = nn.BatchNorm2d(128)
    self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=(1, 2), padding=1)
    self.bn6 = nn.BatchNorm2d(256)

    # variable to store the shape of the output tensor before flattening the ft.
    # it will be used in decoder to reconstruct
    self.shape_before_flatten = (256, 4, 4)

    # compute the flattened size after convolutions
    flattened_size = 4*4*256

    self.fc = nn.Linear(flattened_size, embedding_dim)
    self.relu = nn.ReLU()

  def forward(self, x):
    # apply ReLU activations after each convolutional Layer
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.relu(self.bn4(self.conv4(x)))
    x = self.relu(self.bn5(self.conv5(x)))
    x = self.relu(self.bn6(self.conv6(x)))

    # store the shape before flatten
    self.shape_before_flatten = x.shape[1:]

    # flatten the tensor
    x = x.view(x.size(0), -1)

    # apply fully connected layer to generate embeddings
    x = self.fc(x)
    return x


class Decoder(nn.Module):
  def __init__(self, embedding_dim, shape_before_flatten, channels, isUNet=False):
    super(Decoder, self).__init__()

    # define fully connected layer to unflatten the embeddings
    self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flatten))
    # store the shape before flatten
    self.reshape_dim = shape_before_flatten
    self.isUNet = isUNet

    if self.isUNet:
      # define transpose convolutional layers
      self.deconv1 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
      self.bn1 = nn.BatchNorm2d(256)
      self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
      self.bn2 = nn.BatchNorm2d(128)
      self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
      self.bn3 = nn.BatchNorm2d(64)
      self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
      self.bn4 = nn.BatchNorm2d(64)
      self.deconv5 = nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
      self.bn5 = nn.BatchNorm2d(32)

      # final conv layer to reduce the channel dimension to match the number of output channels
      self.conv1 = nn.Conv2d(32, channels, kernel_size=1)


    else:

      # define transpose convolutional layers
      self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=(1, 2),
                                        padding=1, output_padding=(0, 1))
      self.bn1 = nn.BatchNorm2d(256)
      self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=(2, 2),
                                        padding=1, output_padding=1)
      self.bn2 = nn.BatchNorm2d(128)
      self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(2, 2),
                                        padding=1, output_padding=1)
      self.bn3 = nn.BatchNorm2d(64)
      self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=(2, 2), #128
                                        padding=2, output_padding=(1, 1))   #(2,1) (1,0)
      self.bn4 = nn.BatchNorm2d(32)
      self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=(2, 2),
                                        padding=2, output_padding=(1, 1)) #(2,1) (1,0)
      self.bn5 = nn.BatchNorm2d(16)

      self.deconv6 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=(2, 2),
                                        padding=2, output_padding=(1, 1))  # (2,1) (1,0)
      self.bn6 = nn.BatchNorm2d(8)
      # final conv layer to reduce the channel dimension to match the number of output channels
      self.conv1 = nn.Conv2d(8, channels, kernel_size=1)

    # define final convolutional layer to generate output image
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
    x = self.relu(self.bn6(self.deconv6(x)))

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


class UNet(nn.Module):
  def __init__(self, input_shape, embedding_dim):
    super(UNet, self).__init__()
    self.shape_before_flatten = (512, 4, 4)
    self.Encoder = Encoder(input_shape, input_shape[0], embedding_dim)
    self.Decoder = Decoder(embedding_dim, self.shape_before_flatten, input_shape[0], 2)

  def forward(self, x):
    # store the outputs of each encoding layer
    enc1 = self.Encoder.relu(self.Encoder.bn1(self.Encoder.conv1(x)))
    enc2 = self.Encoder.relu(self.Encoder.bn2(self.Encoder.conv2(enc1)))
    enc3 = self.Encoder.relu(self.Encoder.bn3(self.Encoder.conv3(enc2)))
    enc4 = self.Encoder.relu(self.Encoder.bn4(self.Encoder.conv4(enc3)))
    enc5 = self.Encoder.relu(self.Encoder.bn5(self.Encoder.conv5(enc4)))
    self.shape_before_flatten = enc5.shape[1:]

    # pass the last encoding layer's output through the fully connected layer
    features = self.Encoder.fc(enc5.view(enc5.size(0), -1))

    # start decoding and add skip connections
    features = self.Decoder.fc(features)
    features = features.view(features.size(0), *self.shape_before_flatten)

    features = torch.cat((features, enc5), dim=1)  # skip connection
    dec1 = self.Decoder.relu(
      self.Decoder.bn1(self.Decoder.deconv1(features)))

    dec1 = torch.cat((dec1, enc4), dim=1)  # skip connection
    dec2 = self.Decoder.relu(self.Decoder.bn2(self.Decoder.deconv2(dec1)))

    dec2 = torch.cat((dec2, enc3), dim=1)  # skip connection
    dec3 = self.Decoder.relu(self.Decoder.bn3(self.Decoder.deconv3(dec2)))

    dec3 = torch.cat((dec3, enc2), dim=1)  # skip connection
    dec4 = self.Decoder.relu(self.Decoder.bn4(self.Decoder.deconv4(dec3)))

    dec4 = torch.cat((dec4, enc1), dim=1)  # skip connection
    dec5 = self.Decoder.relu(self.Decoder.bn5(self.Decoder.deconv5(dec4)))

    # generate output image
    x = self.Decoder.sigmoid(self.Decoder.conv1(dec5))

    return x


