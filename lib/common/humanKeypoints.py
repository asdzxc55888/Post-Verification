import numpy as np
import pandas as pd
import os

from config import cfg

CORRELATION_PARENT = [None, 0, 0, 0, 0, 0, 0, 5, 6, 5, 6, 0, 0, 11, 12, 11, 12]

LIMBS_LEFT_INDEX = [7, 9, 13, 15]
LIMBS_RIGHT_INDEX = [8, 10, 14, 16]

NOISE_THRESHOLD = 1.2


class humanKeypoints:
    def __init__(self, video_name) -> None:
        self.__video_name = video_name
        self.__milestone = int(str.split(video_name.split('.')[0], "_")[1])
        self.__keypoints_df = self.__initKeypoints()
        self.__origin_keypoints_df = self.__initKeypoints()
        self.__normalizeKeypoints()
        self.__correlations = self.__initCorrelations()
        self.__noise_frames = None
        self.__noise_frames = self.getNoiseFrames(NOISE_THRESHOLD)

    def __initKeypoints(self) -> pd.DataFrame:
        '''
        Keypoints info from the csv file.

        Return
        ---
        keypoints : pandas DataFrame
        '''
        keypoints_df = pd.read_csv(os.path.join(
            cfg.KEYPOINT_RESULT_PATH, self.__video_name.split('.')[0] + ".csv"))
        # transfer string to array
        keypoints_df['keypoints'] = keypoints_df['keypoints'].apply(eval)
        return keypoints_df

    def __initCorrelations(self) -> np.ndarray:
        correlations = []
        for i in range(len(self.__keypoints_df.index)):
            keypoints = np.asarray(self.__keypoints_df['keypoints'].values[i])
            correlations.append(self.__computeCorrelation(keypoints))

        return np.asarray(correlations)

    def __computeCorrelation(self, keypoints: list):
        keypoints_correlation = []
        for i in range(len(CORRELATION_PARENT)):
            if CORRELATION_PARENT[i] is None:
                keypoints_correlation.append(keypoints[i])
            else:
                parent_keypoint = keypoints[CORRELATION_PARENT[i]]
                correlation_position = keypoints[i] - parent_keypoint
                keypoints_correlation.append(correlation_position)
        return keypoints_correlation

    def __normalizeKeypoint(self, keypoints, xmin, ymin, width, heigh):
        '''
        Normalize keypoints of single frame
        '''
        for i in range(len(keypoints)):
            keypoints[i][0] = (keypoints[i][0] - xmin) / width
            keypoints[i][1] = (keypoints[i][1] - ymin) / width
        return keypoints

    def __normalizeKeypoints(self) -> None:
        '''
        Normalize keypoints.
        '''
        bbox_info = pd.read_csv(os.path.join(
            cfg.BBOX_INFO_PATH, self.__video_name.split(".")[0] + ".csv"))
        boxes = bbox_info["boxes"]
        right_top = []
        left_bottom = []
        for i in range(len(boxes)):
            right_top.append(list(eval(boxes[i])[0]))
            left_bottom.append(list(eval(boxes[i])[1]))
        right_top = np.asarray(right_top)
        left_bottom = np.asarray(left_bottom)

        xmin = right_top[..., 0].min()
        ymin = right_top[..., 1].min()
        width = left_bottom[..., 0].max() - xmin
        height = left_bottom[..., 1].max() - ymin

        self.__keypoints_df['keypoints'] = self.__keypoints_df['keypoints'].apply(
            lambda x: self.__normalizeKeypoint(x, xmin, ymin, width, height))

    def __removeVariabilityNoise(self, variability: list) -> list:
        noise_frame = self.getNoiseFrames(threshold=NOISE_THRESHOLD)
        for i in range(len(variability)):
            if noise_frame[i]:
                variability[i] = 0
        return variability

    def __getMaximumLimbsLenght(self):
        correlation = []
        noise_frame = self.__noise_frames
        for i in range(len(noise_frame)):
            if not noise_frame[i]:
                correlation.append(self.__correlations[i])
        correlation = np.asarray(correlation)

        distance = np.linalg.norm(correlation, axis=2)
        return np.max(distance, axis=0)

    def getNoiseFrames(self, threshold) -> list:
        '''
        Get frame index of video which predicted error.
        Return
        ---
        noise_frame : `int` list
        '''
        variablility = self.getKeypointVariability(isRemoveNoise=False)
        v_sum = np.sum(variablility, axis=1)
        return v_sum > threshold

    def getVideoName(self) -> str:
        '''
        Return
        ---
        video_name: video name of keypoints.
        '''
        return self.__video_name

    def getMilestone(self):
        '''
        Return
        ---
        milestion: `int` milestone of infant.
        '''
        return self.__milestone

    def getKeypoints(self, isNormalize=True) -> pd.DataFrame:
        '''
        Keypoints of predicted result.

        Return
        ---
        keypoint: pandas DataFrame
        '''
        if isNormalize:
            return self.__keypoints_df
        else:
            return self.__origin_keypoints_df

    def getKeypointVariability(self, isRemoveNoise=True, isAddedVertical=True) -> np.ndarray:
        '''
        Variability of keypoints.

        Return
        ---
        variability : `float` ndarray
        '''
        result = np.linalg.norm(np.diff(self.__correlations, axis=0), axis=2)
        if self.__noise_frames is not None and isAddedVertical:
            max_length = self.__getMaximumLimbsLenght()
            max_limbs_length = np.max([max_length[LIMBS_LEFT_INDEX],
                                       max_length[LIMBS_RIGHT_INDEX]], axis=0)
            correlations_distance = np.linalg.norm(self.__correlations, axis=2)
            limbs_left = max_limbs_length - correlations_distance[..., LIMBS_LEFT_INDEX]
            limbs_left_diff = np.diff(limbs_left, axis=0)
            limbs_right = max_limbs_length - correlations_distance[..., LIMBS_RIGHT_INDEX]
            limbs_right_diff = np.diff(limbs_right, axis=0)
            result[..., LIMBS_LEFT_INDEX] += limbs_left_diff
            result[..., LIMBS_RIGHT_INDEX] += limbs_right_diff

        if isRemoveNoise:
            return self.__removeVariabilityNoise(result)
        else:
            return result

    def getAverageMovement(self, eachAverageTime: int = 24) -> np.ndarray:
        '''
        Average movement distance of the variability of the keypoints.

        Parameter
        ---
        eachAverageTime : each average time, default is 24.

        Return
        ---
        average_movement : ndarray
        '''
        result = []
        variability = self.getKeypointVariability()
        for joint_index in range(variability.shape[1]):
            single_result = np.convolve(variability[..., joint_index], np.ones(eachAverageTime), 'valid') / eachAverageTime
            result.append(single_result)
        result = np.asarray(result)
        return result

    def getFeature(self):
        return None
