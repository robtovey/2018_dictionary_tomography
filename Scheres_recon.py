'''
Created on 25 Jun 2018

@author: Rob Tovey
'''
from os.path import join
from odl.contrib import mrc
from matplotlib import pyplot as plt
from numpy import random, arange
from scipy.io import savemat


# with mrc.FileReaderMRC(join('PolyII', 'tiltseries_nonoise.mrc')) as f:
with mrc.FileReaderMRC(join('store', 'CtfSignalOnly.mrcs')) as f:
    f.read_header()
    # pixels-x, pixels-y, projections-z
    shape = f.data_shape
    # offset (bytes) due to header
    header = f.header_size
    # length of one float (bytes)
    pix_size = f.data_dtype.itemsize
    # length of one image (bytes)
    im_size = pix_size * shape[0] * shape[1]

    random.seed(0)
    slices = random.choice(arange(shape[2]), 100, False)

    data = [f.read_data(header + i * im_size, header + (i + 1) * im_size, False)
            for i in slices]

# with mrc.FileReaderMRC(join('PolyII', 'tiltseries.mrc')) as f:
with mrc.FileReaderMRC(join('store', 'CtfSignal_Plus_1xNoise.mrcs')) as f:
    f.read_header()
    # pixels-x, pixels-y, projections-z
    shape = f.data_shape
    # offset (bytes) due to header
    header = f.header_size
    # length of one float (bytes)
    pix_size = f.data_dtype.itemsize
    # length of one image (bytes)
    im_size = pix_size * shape[0] * shape[1]

    noisy = [f.read_data(header + i * im_size, header + (i + 1) * im_size, False)
             for i in slices]

for i in range(len(slices)):
    savemat(join('store', 'proj' + str(slices[i]) + '.mat'),
            {'gt': data[i], 'noisy': noisy[i]})

for i in range(len(slices)):
    plt.subplot('121').clear()
    plt.imshow((data[i].T))
    plt.subplot('122').clear()
    plt.imshow((noisy[i].T))
    plt.title('Number ' + str(slices[i]))
    plt.show(block=False)
    plt.pause(.1)
plt.show()

print('Finished')
