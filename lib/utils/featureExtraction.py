import math
from pyexpat import features
from common.humanKeypoints import humanKeypoints
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd

FPS = 24
DOWNSAMPLE = 2
MOVING_AVERAGE_TIME_PERIOD = FPS/DOWNSAMPLE * 5

def extractFeatures(infantKeypoints: humanKeypoints):
    moving_average = infantKeypoints.getAverageMovement(eachAverageTime=120)
    moving_average = np.delete(moving_average, [1, 2, 3, 4], axis=0)

    split_times = math.floor(moving_average.shape[1] / (FPS * 90))

    # features = []
    mean_moving_average = []
    max_moving_average = []
    min_moving_average = []
    std_moving_average = []
    skew_moving_average = []
    kur_moving_average = []
    for i in range(split_times):
        start = i * (FPS * 30)
        end = (i+1) * (FPS * 30)
        ma_range = moving_average[..., start:end]
        mean_moving_average.append(np.mean(ma_range, axis=1))
        max_moving_average.append(np.max(ma_range, axis=1))
        min_moving_average.append(np.min(ma_range, axis=1))
        std_moving_average.append(np.std(ma_range, axis=1))
        skew_moving_average.append(skew(ma_range, axis=1))
        kur_moving_average.append(kurtosis(ma_range, axis=1))

    # mean_moving_average = np.mean(moving_average, axis=1)
    # max_moving_average = np.max(moving_average, axis=1)
    # min_moving_average = np.min(moving_average, axis=1)
    # std_moving_average = np.std(moving_average, axis=1)
    # skew_moving_average = skew(moving_average, axis=1)
    # kur_moving_average = kurtosis(moving_average, axis=1)

    features = {
        "mean_moving_average": mean_moving_average,
        "max_moving_average": max_moving_average,
        "min_moving_average": min_moving_average,
        "std_moving_average": std_moving_average,
        "skew_moving_average": skew_moving_average,
        "kur_moving_average": kur_moving_average,
        "milestone": infantKeypoints.getMilestone()
    }
    df = pd.DataFrame(features)
    return df
