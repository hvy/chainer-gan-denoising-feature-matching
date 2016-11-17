import numpy as np
import chainer
from chainer import training, reporter
from chainer import functions as F
from chainer import Variable


class GenerativeAdversarialUpdater(training.StandardUpdater):
    def __init__(self, *, iterator, noise_iterator, optimizer_generator,
                 optimizer_discriminator, optimizer_denoiser, lambda_denoise,
                 lambda_adv, device=-1):

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'gen': optimizer_generator,
                      'dis': optimizer_discriminator,
                      'r': optimizer_denoiser}

        super().__init__(iterators, optimizers, device=device)

        self.lambda_denoise = lambda_denoise
        self.lambda_adv = lambda_adv

        if device >= 0:
            chainer.cuda.get_device(device).use()
            [optimizer.target.to_gpu() for optimizer in optimizers.values()]

        self.xp = chainer.cuda.cupy if device >= 0 else np
        self.report = True

    @property
    def generator(self):
        return self._optimizers['gen'].target

    @property
    def discriminator(self):
        return self._optimizers['dis'].target

    @property
    def denoiser(self):
        return self._optimizers['r'].target

    @property
    def generator_optimizer(self):
        return self._optimizers['gen']

    @property
    def discriminator_optimizer(self):
        return self._optimizers['dis']

    @property
    def denoiser_optimizer(self):
        return self._optimizers['r']

    @property
    def feature_extractor(self):
        return self._optimizers['dis'].target.feature_extractor

    def update_core(self):
        z = self.converter(self._iterators['z'].next(), self.device)
        x_real = self.converter(self._iterators['main'].next(), self.device)

        loss_discriminator = self.forward_discriminator(z, x_real)
        loss_generator = self.forward_generator(z)
        loss_denoiser = self.forward_denoiser(x_real)

        if self.report:
            reporter.report({'dis/loss': loss_discriminator})
            reporter.report({'gen/loss': loss_generator})
            reporter.report({'denoiser/loss': loss_denoiser})

        self.discriminator.cleargrads()
        loss_discriminator.backward()
        self.discriminator_optimizer.update()

        self.generator.cleargrads()
        loss_generator.backward()
        self.generator_optimizer.update()

        self.denoiser.cleargrads()
        loss_denoiser.backward()
        self.denoiser_optimizer.update()

    def forward_discriminator(self, z, x_real):
        x_fake = self.generator(Variable(z))
        y_fake = self.discriminator(x_fake)
        y_real = self.discriminator(Variable(x_real))

        zeros = Variable(self.xp.zeros(y_fake.data.shape[0], dtype=self.xp.int32))
        ones = Variable(self.xp.ones(y_fake.data.shape[0], dtype=self.xp.int32))
        loss = F.softmax_cross_entropy(y_fake, zeros)
        loss += F.softmax_cross_entropy(y_real, ones)

        return loss

    def forward_generator(self, z):
        x_fake = self.generator(Variable(z))
        y_fake, features = self.discriminator(x_fake, features=True)
        denoised = self.denoiser(features)

        ones = Variable(self.xp.ones(y_fake.data.shape[0], dtype=self.xp.int32))
        loss = self.lambda_adv * F.softmax_cross_entropy(y_fake, ones)
        loss += self.lambda_denoise * F.mean_squared_error(features, denoised)

        return loss

    def forward_denoiser(self, x_real):
        features = self.feature_extractor(Variable(x_real))

        # Corruption function, i.e. add Gaussian noise to the features
        # TODO: Anneal the std towards 0 according to paper
        mean = Variable(self.xp.zeros_like(features.data, dtype=self.xp.float32))
        ln_var = Variable(self.xp.zeros_like(features.data, dtype=self.xp.float32))
        noise = F.gaussian(mean, ln_var)
        corrupted_features = features + noise

        # NOTE: In the original paper, the authors train a denoising autoencoder
        # that learns to reconstructor the corruption, this implementation
        # is different in the sense that the autoencoder actually learns to
        # remove the noiser
        denoised = self.denoiser(corrupted_features)

        loss = F.mean_squared_error(features, denoised)

        return loss

    def sample(self):
        """Return a sample from the generator with random noise."""
        z_it = self._iterators['z'].next()
        z = self.converter(z_it, self.device)

        x_fake = self.generator(Variable(z), test=True)

        return x_fake
