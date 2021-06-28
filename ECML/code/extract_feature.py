import numpy as np
import pandas as pd
import tsfresh as tsf


def common_feature(npy):
    feature_average = np.average(npy, axis=1)
    feature_std = np.std(npy, axis=1)
    return np.vstack((feature_average, feature_std)).T


def caculate_feature(x, mode):
    x = pd.Series(x)
    param_coeff = [{'coeff': 1, 'attr': 'real'}]
    param_energy = [{'num_segments': 10, 'segment_focus': 4}]
    param_spkt = [{'coeff': 2}]
    param_agg = [{'f_agg': 'mean', 'maxlag':40}]
    if (mode == 0):
        return list(tsf.feature_extraction.feature_calculators.fft_coefficient(x, param_coeff))[0][1]
    elif (mode == 1):
        return list(tsf.feature_extraction.feature_calculators.energy_ratio_by_chunks(x, param_energy))[0][1]
    elif (mode == 2):
        return list(tsf.feature_extraction.feature_calculators.spkt_welch_density(x, param_spkt))[0][1]
    elif (mode == 3):
        return list(tsf.feature_extraction.feature_calculators.agg_autocorrelation(x, param_agg))[0][1]


def time_feature(npy):
    feature_coeff  = np.apply_along_axis(caculate_feature, 1, npy, 0)
    feature_energy = np.apply_along_axis(caculate_feature, 1, npy, 1)
    feature_spkt   = np.apply_along_axis(caculate_feature, 1, npy, 2)
    feature_agg    = np.apply_along_axis(caculate_feature, 1, npy, 3)
    features = np.vstack((feature_coeff, feature_energy, feature_spkt, feature_agg)).T
    return features


def combine_feature(npy):
    feature_common = common_feature(npy[:, 7:])
    feature_time = time_feature(npy[:, 7:])
    features = np.hstack((npy[:, :7], feature_common, feature_time))
    return features
