import numpy
from chainer.dataset import iterator


def to_tuple(x):
    if hasattr(x, '__getitem__'):
        return x
    return x,


class UniformNoiseGenerator(object):
    def __init__(self, low, high, size):
        self.low = low
        self.high = high
        self.size = to_tuple(size)

    def __call__(self, batch_size):
        return numpy.random.uniform(self.low, self.high, (batch_size,) +
                                    self.size).astype(numpy.float32)


class GaussianNoiseGenerator(object):
    def __init__(self, loc, scale, size):
        self.loc = loc
        self.scale = scale
        self.size = to_tuple(size)

    def __call__(self, batch_size):
        return numpy.random.normal(self.loc, self.scale, (batch_size,) +
                                   self.size).astype(numpy.float32)


class RandomNoiseIterator(iterator.Iterator):
    def __init__(self, noise_generator, batch_size):
        self.noise_generator = noise_generator
        self.batch_size = batch_size

    def __next__(self):
        batch = self.noise_generator(self.batch_size)
        return batch
