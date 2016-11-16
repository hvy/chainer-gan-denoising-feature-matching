import chainer
from chainer import Chain, Variable
from chainer import functions as F
from chainer import links as L


class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__(
            fc=L.Linear(None, 1024),
            dc1=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dc3=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            dc4=L.Deconvolution2D(32, 3, 4, stride=2, pad=1),
            bn1=L.BatchNormalization(1024),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
            bn4=L.BatchNormalization(32))

    def __call__(self, z, test=False):
        h = F.relu(self.bn1(self.fc(z), test=test))
        h = F.reshape(h, (z.shape[0], 256, 2, 2))
        h = F.relu(self.bn2(self.dc1(h), test=test))
        h = F.relu(self.bn3(self.dc2(h), test=test))
        h = F.relu(self.bn4(self.dc3(h), test=test))
        h = F.sigmoid(self.dc4(h))
        return h


class Discriminator(Chain):
    def __init__(self):
        super().__init__(
            feature_extractor=FeatureExtractor(),
            classifier=Classifier())

    def __call__(self, x, test=False):
        h = self.feature_extractor(x, test=test)
        h = self.classifier(h)
        return h


class FeatureExtractor(Chain):
    """Feature extractor, phi."""
    def __init__(self):
        super().__init__(
            c1=L.Convolution2D(None, 32, 4, stride=2, pad=1),
            c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            c3=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            c4=L.Convolution2D(128, 256, 4, stride=2, pad=1),
            fc=L.Linear(None, 512),
            bn1=L.BatchNormalization(64),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(256))

    def __call__(self, x, test=False, corrupt=False):
        h = F.leaky_relu(self.c1(x))
        h = F.leaky_relu(self.bn1(self.c2(h), test=test))
        h = F.leaky_relu(self.bn2(self.c3(h), test=test))
        h = F.leaky_relu(self.bn3(self.c4(h), test=test))
        h = F.leaky_relu(self.fc(h))

        if corrupt:
            # Add Gaussian noise. This code should run during the denoiser
            # training forward pass
            mean = self.xp.zeros_like(h.data, dtype=self.xp.float32)
            ln_var = self.xp.zeros_like(h.data, dtype=self.xp.float32)
            h += F.gaussian(mean=Variable(mean), ln_var=Variable(ln_var))

        return h


class Classifier(Chain):
    """Real/Fake classifier used in the discriminator."""
    def __init__(self):
        super().__init__(
            fc=L.Linear(None, 2))

    def __call__(self, x, corrupt=False):
        h = self.fc(x)
        return h


class Denoiser(Chain):
    """Denoiser, r."""
    # TODO: Add batch normalization according to paper.
    def __init__(self):
        super().__init__(
            l1=L.Linear(None, 512),
            l2=L.Linear(512, 512),
            l3=L.Linear(512, 512),
            l4=L.Linear(512, 512),
            l5=L.Linear(512, 512))

    def __call__(self, k, test=False):
        # TODO: Reconsider activation functions to match the feature extractor
        h = F.relu(self.l1(k))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = F.relu(self.l5(h))
        return h
