#!/usr/bin/env python3
# Based on https://gist.github.com/akesling/5358964

import os
import urllib
import shutil
import gzip

import struct
import numpy as np

MNIST_BASEURL = 'http://yann.lecun.com/exdb/mnist'
MNIST_FILES = {
    'training': {
        'images': 'train-images-idx3-ubyte',
        'labels': 'train-labels-idx1-ubyte',
    },
    'testing': {
        'images': 't10k-images-idx3-ubyte',
        'labels': 't10k-labels-idx1-ubyte',
    },
}

def download():
    for dataset in ('training', 'testing'):
        for kind in ('images', 'labels'):
            name = MNIST_FILES[dataset][kind]
            _download_dataset(name)

def read(dataset, path='.'):
    if dataset not in ('training', 'testing'):
        raise ValueError("dataset must be 'training' or 'testing'")

    labels_filename = os.path.join(path, MNIST_FILES[dataset]['labels'])
    images_filename = os.path.join(path, MNIST_FILES[dataset]['images'])

    with open(labels_filename, 'rb') as labels_fp:
        magic, num = struct.unpack('>II', labels_fp.read(8))
        labels = np.fromfile(labels_fp, dtype=np.int8)

    with open(images_filename, 'rb') as images_fp:
        magic, num, rows, cols = struct.unpack('>IIII', images_fp.read(16))
        images = np.fromfile(images_fp, dtype=np.uint8).reshape(len(labels), rows, cols)

    for i, label in enumerate(labels):
        yield (label, images[i])


def show(image):
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=matplotlib.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()

def show_ascii(image):
    for y in image:
        row = ''
        for x in y:
            row += '{0: <4}'.format(x)
        print(row)

# -----------------------------------------------------------------------------

def _download_dataset(name):
    basedir = os.path.dirname(__file__)
    filename = os.path.join(basedir, name)
    filename_gz = '{0}.gz'.format(filename)
    url = "{baseurl}/{name}.gz".format(baseurl=MNIST_BASEURL, name=name)

    if os.path.exists(filename):
        print("Found existing file {0}".format(filename))
        return

    print("Downloading {0}".format(url))
    with urllib.request.urlopen(url) as response, open(filename_gz, 'wb') as fp_gz:
        shutil.copyfileobj(response, fp_gz)

    with gzip.open(filename_gz, 'rb') as fp_gz, open(filename, 'wb') as fp:
        shutil.copyfileobj(fp_gz, fp)

    os.remove(filename_gz)

# EOF
