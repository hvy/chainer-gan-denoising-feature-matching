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

    def forward(self, test=False, report=True):
        z_it = self._iterators['z'].next()
        z = self.converter(z_it, self.device)

        x_fake = self.generator(Variable(z), test=test)
        y_fake = self.discriminator(x_fake, generated=True, test=test)

        x_real_it = self._iterators['main'].next()
        x_real = self.converter(x_real_it, self.device)

        y_real = self.discriminator(Variable(x_real), generated=False)


        if test:
            return x_fake
        else:
            return y_fake, y_real

    def update_params(self, losses, report=True):
        for name, loss in losses.items():
            if report:
                reporter.report({'{}/loss'.format(name): loss})

            self._optimizers[name].target.cleargrads()
            loss.backward()
            self._optimizers[name].update()

    def __update_core(self):
        if self.is_new_epoch:
            pass

        losses = self.backward(self.forward())
        self.update_params(losses, report=True)

    def update_discriminator(self, z, x_real):
        x_fake = self.generator(Variable(z))
        y_fake = self.discriminator(x_fake, generated=True)
        y_real = self.discriminator(Variable(x_real), generated=False)

        zeros = Variable(self.xp.zeros(y_fake.shape[0], dtype=self.xp.int32))
        ones = Variable(self.xp.ones(y_fake.shape[0], dtype=self.xp.int32))
        loss = F.softmax_cross_entropy(y_fake, zeros)
        loss += F.softmax_cross_entropy(y_real, ones)

        self.discriminator.cleargrads()
        loss.backward()
        self.discriminator_optimizer.update()

        return loss

    def update_generator(self, z):
        x_fake = self.generator(Variable(z))
        y_fake, features = self.discriminator(x_fake, generated=True, features=True)
        denoised = self.denoiser(features, generated=True)

        ones = Variable(self.xp.ones(y_fake.shape[0], dtype=self.xp.int32))
        loss = self.lambda_adv * F.softmax_cross_entropy(y_fake, ones)
        loss = self.lambda_denoise * F.mean_squared_error(features, denoised)

        self.generator.cleargrads()
        loss.backward()
        self.generator_optimizer.update()

        return loss

    def update_denoiser(self, x_real):
        features = self.feature_extractor(Variable(x_real), generated=False)

        # Corruption function, i.e. add Gaussian noise to the features
        #
        # TODO: Anneal the std towards 0 according to paper
        #
        # NOTE: In the original paper, the authors train a denoising autoencoder
        # that learns to reconstructor the corruption, this implementation
        # is different in the sense that the autoencoder actually learns to
        # remove the noiser
        mean = self.xp.zeros_like(features.data, dtype=self.xp.float32)
        ln_var = self.xp.zeros_like(features.data, dtype=self.xp.float32)
        noise = F.gaussian(mean=Variable(mean), ln_var=Variable(ln_var))
        corrupted_features = features + noise

        denoised = self.denoiser(corrupted_features, generated=False)
        loss = F.mean_squared_error(features, denoised)

        self.denoiser.cleargrads()
        loss.backward()
        self._optimizers['r'].update()

        return loss

    def update_core(self):
        # Update all networks once each, you may update any network more than
        # once to balance the training in this iteration
        z = self.converter(self._iterators['z'].next(), self.device)
        x_real = self.converter(self._iterators['main'].next(), self.device)

        loss_discriminator = self.update_discriminator(z, x_real)
        loss_generator = self.update_generator(z)
        loss_denoiser = self.update_denoiser(x_real)

        if self.report:
            reporter.report({'dis/loss': loss_discriminator})
            reporter.report({'gen/loss': loss_generator})
            reporter.report({'denoiser/loss': loss_denoiser})
