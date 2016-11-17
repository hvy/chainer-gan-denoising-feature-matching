# GAN with Denoising Feature Matching

An unofficial attempt to implement the GAN proposed in [Improving Generative Adversarial Networks with Denoising Feature Matching](http://openreview.net/forum?id=S1X7nhsxl) using Chainer.

Details such as activations and hyperparameters not presented in the paper are simply guessed.

The denoising autoencoder in the original papers is trained to reconstructs corrupted images with Gaussian noise. **In this implementation the autoencoder is trained to remove the noise instead**. *Edit: According to the author, this is simply a typo in the paper and the autoencoder should be trained to remove the noise as in this implementation.*

This implementation does not separately keep track of the batch normalization statistics for the discrimnator (including the feature extractor) and the denoising autoencoder for real and generated data.

The corruption function used when updating the parameters of the autoencoder is not annealed.

### Loss

The network is trained on 32x32 RGB images (3 channels) from CIFAR-10.

<img src="./samples/log.png" width="512px;"/>

- **Discriminator Loss** Traditional discriminator GAN loss.

- **Generator Loss** Traditional generator GAN loss and reconstruction error (L2, mean squared error).

- **Denoiser Loss** Denoising autoencoder reconstruction error (L2, mean squared error).

### Samples

**Epoch 100**

<img src="./samples/100.png" width="512px;"/>

**Epoch 80**

<img src="./samples/80.png" width="512px;"/>

**Epoch 60**

<img src="./samples/60.png" width="512px;"/>

**Epoch 40**

<img src="./samples/40.png" width="512px;"/>

**Epoch 20**

<img src="./samples/20.png" width="512px;"/>

**Epoch 1**

<img src="./samples/1.png" width="512px;"/>
