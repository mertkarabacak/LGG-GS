import random
from scipy import ndimage
import numpy as np

x_train = np.load("np/x_train.npy")
y_train = np.load("np/y_train.npy")


def rotate(volume):
    """Rotate the volume by a few degrees"""
    # define some rotation angles
    angles = [-80, -40, -20, -10, -5, 5, 10, 20, 40, 80]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, reshape=False)
    volume[volume < 0] = 0
    volume[volume > 1] = 1
    return volume


rotated = (rotate(x_train[0]))
ind = 5*y_train.shape[0];

for i in range(ind):
    x_train = np.concatenate((x_train, np.expand_dims(rotate(x_train[i]), axis=0)), axis=0)
    y_train = np.concatenate((y_train, np.expand_dims(y_train[i], axis=0)), axis=0)
    if i % 10 == 0:
        print(str(i) + "/" + str(ind))

np.save("np/x_train.npy", x_train)
np.save("np/y_train.npy", y_train)
