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

    def __call__(self, x, test=False, features=False):
        f = self.feature_extractor(x, test=test)
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
            fc=L.Linear(None, 512),
            bn_c2=L.BatchNormalization(64),
            bn_c3=L.BatchNormalization(128),
            bn_c4=L.BatchNormalization(256))

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c1(x))
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
    def __init__(self):
        super().__init__(
            l1=L.Linear(512, 512),
            l2=L.Linear(512, 512),
            l3=L.Linear(512, 512),
            l4=L.Linear(512, 512),
            l5=L.Linear(512, 512),
            bn_l2=L.BatchNormalization(512),
            bn_l3=L.BatchNormalization(512),
            bn_l4=L.BatchNormalization(512))

    def __call__(self, x, test=False):
        h = F.relu(self.l1(x))
        h = F.relu(self.bn_l2(self.l2(h), test=test))
        h = F.relu(self.bn_l3(self.l3(h), test=test))
        h = F.relu(self.bn_l4(self.l4(h), test=test))
        h = self.l5(h)
        return h
