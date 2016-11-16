import argparse
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='result/log')
    parser.add_argument('--out', type=str, default='result/log.png')
    parser.add_argument('--keys', nargs='+', type=str, default=['dis/loss', 'gen/loss', 'den/loss'])
    return parser.parse_args()


def load_log(filename, keys):
    """Parse a JSON file and return a dictionary with the given keys. Each
    key maps to a list of corresponding data measurements in the file."""
    log = collections.defaultdict(list)
    with open(filename) as f:
        for data in json.load(f):  # For each type of data
            for key in keys:
                log[key].append(data[key])
    return log


def plot_log(filename, log):
    """Create a plot from the given log and write it to disk as an image."""
    for key, data in log.items():
        plt.plot(range(len(data)), data, label=key)

    ax = plt.gca()
    # ax.set_ylim([0, 2])
    ax.legend(loc='best')
    plt.savefig(filename)
    plt.clf()
    plt.close()


def main(args):
    log = load_log(args.log, args.keys)
    plot_log(args.out, log)


if __name__ == '__main__':
    args = parse_args()
    main(args)
