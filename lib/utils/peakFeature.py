import numpy as np
from scipy.signal import find_peaks

from config import cfg


class peakFeature:
    def __init__(self, all_moving_average, video_name) -> None:
        self.__video_name = video_name
        self.__low_standard = self.__initLowStandard(all_moving_average)
        self.__all_moving_average = self.__initMovingAverage(
            all_moving_average)
        self.__peaks = self.__findPeaks()
        self.__segments = list(map(lambda i: self.__findPeakSegments(
            self.__low_standard[i], self.__all_moving_average[i]), np.arange(self.__all_moving_average.shape[0])))

    def __initLowStandard(self, all_moving_average: np.ndarray) -> np.ndarray:
        return list(map(lambda x: np.mean(x[x < np.percentile(x, 50)]), all_moving_average))

    def __initMovingAverage(self, all_moving_average):
        result = []

        for i, ma in enumerate(all_moving_average):
            ma[ma < self.__low_standard[i]] = self.__low_standard[i]
            result.append(self.__removedMovingAverageNoise(
                self.__low_standard[i], ma))
        return np.array(result)

    def __findPeaks(self) -> list:
        result = []
        for ma in self.__all_moving_average:
            percentile = np.percentile(ma, 50)
            highMean = np.mean(ma[ma > percentile])
            peaks, _ = find_peaks(ma, prominence=highMean/10)
            result.append(peaks)
        return result

    def __removedMovingAverageNoise(self, low_standard, moving_average) -> np.ndarray:
        checkRange = int(cfg.MOVING_AVERAGE_TIME_PERIOD)

        result = []
        for i, ma in enumerate(moving_average):
            target_front = moving_average[i:checkRange+i]
            target_back = moving_average[i-checkRange:i]
            front = target_front[target_front == low_standard]
            back = target_back[target_back == low_standard]
            if(front.size != 0 and back.size != 0):
                ma = low_standard
            result.append(ma)
        return np.array(result)

    def __findPeakSegments(self, low_standard, moving_average) -> np.ndarray:
        result = []
        start, end = None, None
        ls = round(low_standard, 10)
        for i, ma in enumerate(moving_average):
            if round(ma, 10) != ls and start == None:
                start = i
            elif round(ma, 10) == ls and end == None and start != None:
                end = i

            if start != None and end != None:
                is_smaller_sampling_range = end - start < cfg.MOVING_AVERAGE_TIME_PERIOD/2
                if not is_smaller_sampling_range:
                    result.append([start, end])
                start, end = None, None
        return np.array(result)

    def getAcceleration(self, moving_average, peaks, segment) -> np.ndarray:
        accelerations = []
        for start, end in segment:
            target = moving_average[start:end]
            peak = peaks[peaks >= start]
            peak = peak[peak <= end]
            if(peak.size == 0):
                peak = np.array([start + np.argmax(target)])

            acceleration = 0
            for j in range(peak.size):
                p = peak[j]
                peak_index = p - start
                if peak_index == 0:
                    acceleration += target[peak_index] / \
                        cfg.MOVING_AVERAGE_TIME_PERIOD
                elif j == 0:
                    acceleration += (target[peak_index] -
                                     target[0]) / peak_index
                else:
                    acceleration += (target[peak_index] - np.min(
                        target[peak[j-1] - start:peak_index])) / (peak[j] - peak[j-1])
            accelerations.append(acceleration)

        return np.array(accelerations)

    def getDeceleration(self, moving_average, peaks, segment) -> np.ndarray:
        decelerations = []
        for start, end in segment:
            target = moving_average[start:end]
            peak = peaks[peaks >= start]
            peak = peak[peak <= end]
            if(peak.size == 0):
                peak = np.array([start + np.argmax(target)])

            deceleration = 0
            for j, p in enumerate(peak):
                peak_index = p - start
                if j == peak.size - 1:
                    deceleration += (target[peak_index] -
                                     target[target.size-1]) / (target.size - peak_index)
                else:
                    deceleration += (target[peak_index] -
                                     target[peak[j+1] - start]) / (peak[j+1] - p)
            decelerations.append(deceleration)

        return np.array(decelerations)

    def __computeEachSegment(self, moving_average, segment, func) -> np.ndarray:
        result = []
        for start, end in segment:
            target = moving_average[start:end]
            result.append(func(target))
        return np.array(result)

    def __getSegmentPeakNum(self, peaks, segment) -> np.ndarray:
        result = []
        for start, end in segment:
            target = peaks[peaks >= start]
            target = target[target <= end]
            result.append(target.shape[0])
        return result

    def getPeakFeatures(self):
        peakFeatures = []
        index = 5
        for moving_average, peak, segment in zip(self.__all_moving_average, self.__peaks, self.__segments):
            features_tuple = {
                "acceleration": self.getAcceleration(moving_average, peak, segment),
                "deceleration": self.getDeceleration(moving_average, peak, segment),
                "movment_duration": list(map(lambda i: segment[i][1] - segment[i][0], np.arange(segment.shape[0]))),
                "mean": self.__computeEachSegment(moving_average, segment, np.mean),
                "std": self.__computeEachSegment(moving_average, segment, np.std),
                "peaks_num": self.__getSegmentPeakNum(peak, segment)
            }
            for i in range(features_tuple["acceleration"].shape[0]):
                feature = {
                    "video_name": self.__video_name,
                    "joint_index": index,
                    "acceleration": features_tuple["acceleration"][i],
                    "deceleration": features_tuple["deceleration"][i],
                    "movment_duration": features_tuple["movment_duration"][i],
                    "mean": features_tuple["mean"][i],
                    "std": features_tuple["std"][i],
                    "peaks_num": features_tuple["peaks_num"][i]
                }
                peakFeatures.append(feature)
            index += 1
        return peakFeatures
