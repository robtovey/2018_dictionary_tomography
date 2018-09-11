'''
Created on 25 Jun 2018

@author: Rob Tovey
'''
from os.path import join
from odl.contrib import mrc
from matplotlib import pyplot as plt
from numpy import random, arange
from scipy.io import savemat

RECORD = False

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

if RECORD:
    for i in range(len(slices)):
        savemat(join('store', 'proj' + str(slices[i]) + '.mat'),
                {'gt': data[i], 'noisy': noisy[i]})


def radial_profile(data):
    from numpy import indices, sqrt, bincount
    center = [t / 2 for t in data.shape]
    y, x = indices((data.shape))
    r = sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = bincount(r.ravel(), data.ravel())
    nr = bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


ax = plt.subplot('121'), plt.subplot('122')
clim = [min((d.min() for d in data)), max((d.max() for d in data))]
print(clim)
for i in range(len(slices)):
    ax[0].clear()
    ax[0].imshow((data[i]), clim=clim)
    ax[1].clear()
    plt.imshow(noisy[i], clim=clim)
    from numpy import fft
#     ax[1].imshow(fft.fftshift(abs(fft.fft2(data[i]))), clim=[0, 300])
#     ax[1].plot(radial_profile(fft.fftshift(abs(fft.fft2(data[i])))))
#     plt.ylim([0, 400])
    plt.title('Number ' + str(slices[i]))
    plt.show(block=False)
    plt.pause(.1)
plt.show()

print('Finished')
