# GAN with Denoising Feature Matching

An unofficial attempt to implement the GAN proposed in [Improving Generative Adversarial Networks with Denoising Feature Matching](http://openreview.net/forum?id=S1X7nhsxl) using Chainer.

### Notes

- This implementation does not separately keep track of the batch normalization statistics for the discrimnator (including the feature extractor) and the denoising autoencoder for real and generated data. This lead to more realistic images.

- The denoising autoencoder in the original papers is trained to reconstructs corrupted images with Gaussian noise. In this implementation the autoencoder is trained to remove the noise instead.
