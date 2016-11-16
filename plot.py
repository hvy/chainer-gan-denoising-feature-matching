import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math


def save_ims(filename, ims, dpi=100):
    n, c, w, h = ims.shape
    x_plots = math.ceil(math.sqrt(n))
    y_plots = x_plots if n % x_plots == 0 else x_plots - 1
    plt.figure(figsize=(w*x_plots/dpi, h*y_plots/dpi), dpi=dpi)

    for i, im in enumerate(ims):
        plt.subplot(y_plots, x_plots, i+1)

        if c == 1:
            plt.imshow(im[0])
        else:
            plt.imshow(im.transpose((1, 2, 0)))

        plt.axis('off')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gray()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,
                            hspace=0)

    plt.savefig(filename, dpi=dpi*2, facecolor='black')
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
