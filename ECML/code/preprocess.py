import numpy as np

def preprocess_X(npy):
    npy_params = npy[:, : 6]
    npy_time = npy[:, 6: ]
    feature_average = np.average(npy_time, axis=1)
    feature_average = np.array([feature_average]).T
    npy_params = np.hstack((npy_params, feature_average))
    npy_params = np.tile(npy_params, 55).reshape((-1, 7))
    npy_time = npy_time.reshape((-1, 300))
    return np.hstack((npy_params, npy_time))


def preporcess_y(npy):
    return npy.flatten()
