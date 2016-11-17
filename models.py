import chainer
from chainer import Chain, Variable
from chainer import functions as F
from chainer import links as L


class Generator(Chain):
    """Generates 32x32 images using four doubling upsamplings on 2x2 images."""
    def __init__(self):
        super(Generator, self).__init__(
            fc=L.Linear(None, 1024),
            dc1=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dc3=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            dc4=L.Deconvolution2D(32, 3, 4, stride=2, pad=1),
            bn_fc=L.BatchNormalization(1024),
            bn_dc1=L.BatchNormalization(128),
            bn_dc2=L.BatchNormalization(64),
            bn_dc3=L.BatchNormalization(32))

    def __call__(self, z, test=False):
        h = F.relu(self.bn_fc(self.fc(z), test=test))
        h = F.reshape(h, (z.shape[0], 256, 2, 2))
        h = F.relu(self.bn_dc1(self.dc1(h), test=test))
        h = F.relu(self.bn_dc2(self.dc2(h), test=test))
        h = F.relu(self.bn_dc3(self.dc3(h), test=test))
        h = F.sigmoid(self.dc4(h))
        return h


class Discriminator(Chain):
    def __init__(self):
        super().__init__(
            feature_extractor=FeatureExtractor(),
            classifier=Classifier())

    def __call__(self, x, generated=False, test=False, features=False):
        f = self.feature_extractor(x, generated=generated, test=test)
        h = self.classifier(f)
        if features:
            return h, f
        return h


class FeatureExtractor(Chain):
    """Feature extractor, phi, that takes an 32x32 images as input and outputs
    an one-dimensional feature vector."""
    def __init__(self):
        super().__init__(
            c1=L.Convolution2D(None, 32, 4, stride=2, pad=1),
            c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            c3=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            c4=L.Convolution2D(128, 256, 4, stride=2, pad=1),
            fc=L.Linear(None, 2048))

        for name in ['c2', 'c3', 'c4']:
            link = getattr(self, name)
            size = getattr(link, 'out_channels')
            self.add_link('bn_{}'.format(name), L.BatchNormalization(size))
            """
            self.add_link('bn_{}_real'.format(name), L.BatchNormalization(size))
            self.add_link('bn_{}_fake'.format(name), L.BatchNormalization(size))
            """

    def __call__(self, x, generated=False, test=False):
        postfix = 'fake' if generated else 'real'
        h = F.leaky_relu(self.c1(x))
        """
        h = F.leaky_relu(getattr(self, 'bn_c2_{}'.format(postfix))(self.c2(h), test=test))
        h = F.leaky_relu(getattr(self, 'bn_c3_{}'.format(postfix))(self.c3(h), test=test))
        h = F.leaky_relu(getattr(self, 'bn_c4_{}'.format(postfix))(self.c4(h), test=test))
        """
        h = F.leaky_relu(self.bn_c2(self.c2(h), test=test))
        h = F.leaky_relu(self.bn_c3(self.c3(h), test=test))
        h = F.leaky_relu(self.bn_c4(self.c4(h), test=test))
        h = self.fc(h)
        return h

class Classifier(Chain):
    def __init__(self):
        super().__init__(fc=L.Linear(None, 2))

    def __call__(self, x):
        h = self.fc(x)
        return h


class Denoiser(Chain):

    def __init__(self, n_layers=10):
        super().__init__()
        self.n_layers = n_layers

        for i in range(self.n_layers):
            self.add_link('l{}'.format(i), L.Linear(None, 2048))

        # Two sets of batch normalization layers for all linear layers except
        # the last one
        for i in range(self.n_layers - 1):
            self.add_link('bn_l{}_real'.format(i), L.BatchNormalization(2048))
            self.add_link('bn_l{}_fake'.format(i), L.BatchNormalization(2048))

    def __call__(self, x, generated=False, test=False):
        postfix = 'fake' if generated else 'real'
        h = x
        for i in range(self.n_layers - 1):
            h = getattr(self, 'l{}'.format(i))(h)
            h = getattr(self, 'bn_l{}_{}'.format(i, postfix))(h, test=test)
            h = F.relu(h)
        h = getattr(self, 'l{}'.format(self.n_layers - 1))(h)

        return h
