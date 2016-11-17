import os
from chainer import training, reporter, cuda
from chainer.training import extension
from lib import imutil


class GeneratorSample(extension.Extension):
    def __init__(self, dirname='sample', sample_format='png'):
        self._dirname = dirname
        self._sample_format = sample_format

    def __call__(self, trainer):
        dirname = os.path.join(trainer.out, self._dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        x = self.sample(trainer)

        filename = '{}.{}'.format(trainer.updater.epoch,
                                  self._sample_format)
        filename = os.path.join(dirname, filename)
        imutil.save_ims(filename, x)

    def sample(self, trainer):
        x = trainer.updater.sample()
        x = x.data
        if cuda.get_array_module(x) == cuda.cupy:
            x = cuda.to_cpu(x)
        return x
