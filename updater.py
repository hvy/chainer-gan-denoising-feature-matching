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
    def feature_extractor(self):
        return self._optimizers['dis'].target.feature_extractor

    @property
    def classifier(self):
        return self._optimizers['dis'].target.classifier

    def forward(self, test=False, report=True):
        z_it = self._iterators['z'].next()
        z = self.converter(z_it, self.device)

        x_fake = self.generator(Variable(z), test=test)
        y_fake = self.discriminator(x_fake, test=test)

        x_real_it = self._iterators['main'].next()
        x_real = self.converter(x_real_it, self.device)

        y_real = self.discriminator(Variable(x_real))


        if test:
            return x_fake
        else:
            return y_fake, y_real

    def backward(self, y):
        y_fake, y_real = y

        generator_loss = F.softmax_cross_entropy(
            y_fake,
            Variable(self.xp.ones(y_fake.shape[0], dtype=self.xp.int32)))
        discriminator_loss = F.softmax_cross_entropy(
            y_fake,
            Variable(self.xp.zeros(y_fake.shape[0], dtype=self.xp.int32)))
        discriminator_loss += F.softmax_cross_entropy(
            y_real,
            Variable(self.xp.ones(y_real.shape[0], dtype=self.xp.int32)))
        discriminator_loss /= 2

        return {'gen': generator_loss, 'dis': discriminator_loss}

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

    def update_discriminator(self, z, x_real, test):
        x_fake = self.generator(Variable(z), test=test)
        y_fake = self.discriminator(x_fake, test=test)
        y_real = self.discriminator(Variable(x_real))

        loss = F.softmax_cross_entropy(y_fake,
            Variable(self.xp.zeros(y_fake.shape[0], dtype=self.xp.int32)))
        loss += F.softmax_cross_entropy(y_real,
            Variable(self.xp.ones(y_real.shape[0], dtype=self.xp.int32)))
        loss /= 2

        return loss

    def update_generator(self, z, test):
        x_fake = self.generator(Variable(z), test=test)
        y_fake = self.discriminator(x_fake, test=test)
        features = self.feature_extractor(x_fake)
        denoised = self.denoiser(features)

        loss = self.lambda_denoise * F.mean_squared_error(features, denoised)
        loss += self.lambda_adv * F.softmax_cross_entropy(y_fake,
            Variable(self.xp.ones(y_fake.shape[0], dtype=self.xp.int32)))

        return loss

    def update_denoiser(self, x_real, test):
        corrupted_features = self.feature_extractor(x_real, test=test, corrupt=True)
        denoised = self.denoiser(corrupted_features)

        loss = F.mean_squared_error(corrupted_features, denoised)

        return loss

    def update_core(self):
        # TODO: Don't hardcode me
        test = False

        z = self.converter(self._iterators['z'].next(), self.device)
        x_real = self.converter(self._iterators['main'].next(), self.device)

        discriminator_loss = self.update_discriminator(z, x_real, test=test)
        generator_loss = self.update_generator(z, test=test)
        denoiser_loss = self.update_denoiser(x_real, test=test)

        reporter.report({'dis/loss': discriminator_loss})
        reporter.report({'gen/loss': generator_loss})
        reporter.report({'denoiser/loss': denoiser_loss})

        self.discriminator.cleargrads()
        discriminator_loss.backward()
        self._optimizers['dis'].update()

        self.generator.cleargrads()
        generator_loss.backward()
        self._optimizers['gen'].update()

        self.denoiser.cleargrads()
        denoiser_loss.backward()
        self._optimizers['r'].update()
