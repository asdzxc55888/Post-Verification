import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import os

from config import cfg


class humanKeypoints:
    def __init__(self, video_name) -> None:
        self.__video_name = video_name
        self.__keypoints_df = self.__initKeypoints()
        self.__origin_keypoints_df = self.__initKeypoints()
        self.__normalizeKeypoints()

    def __initKeypoints(self) -> pd.DataFrame:
        '''
        Keypoints info from the csv file.

        Return
        ---
        keypoints : pandas DataFrame
        '''
        keypoints_df = pd.read_csv(os.path.join(
            cfg.keypoint_result_path, self.__video_name.split('.')[0] + ".csv"))
        # transfer string to array
        keypoints_df['keypoints'] = keypoints_df['keypoints'].apply(eval)
        return keypoints_df

    def __normalizeKeypoint(self, keypoints, xmin, ymin, width, heigh):
        '''
        Normalize keypoints of single frame
        '''
        for i in range(len(keypoints)):
            keypoints[i][0] = (keypoints[i][0] - xmin) / width
            keypoints[i][1] = (keypoints[i][1] - ymin) / heigh
        return keypoints

    def __normalizeKeypoints(self) -> None:
        '''
        Normalize keypoints. 
        '''
        bbox_info = pd.read_csv(os.path.join(
            cfg.video_info_path, self.__video_name.split(".")[0] + ".csv"))
        xmin = bbox_info["xmin"].min()
        ymin = bbox_info["ymin"].min()
        width = bbox_info["xmax"].max() - xmin
        height = bbox_info["ymax"].max() - ymin

        self.__keypoints_df['keypoints'] = self.__keypoints_df['keypoints'].apply(
            lambda x: self.__normalizeKeypoint(x, xmin, ymin, width, height))

    def __removeVariabilityNoise(self, variability: list) -> list:
        mean_variability = np.mean(variability, axis=0)
        threshold = 0.1
        for i in range(len(variability)):
            for j in range(len(variability[i])):
                if variability[i][j] > threshold:
                    variability[i][j] = mean_variability[j]
        return variability

    def getVideoName(self) -> str:
        '''
        Return
        ---
        video_name: video name of keypoints.
        '''
        return self.__video_name

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

    def getKeypointVariability(self, isRemoveNoise=True) -> np.ndarray:
        '''
        Variability of keypoints.

        Return
        ---
        variability : float or ndarray
        '''
        keypoints = []
        for i in range(len(self.__keypoints_df.index)):
            keypoints.append(self.__keypoints_df['keypoints'].values[i])
        keypoints = np.array(keypoints)
        result = np.linalg.norm(np.diff(keypoints, axis=0), axis=2)

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
        for frame_index in range(variability.shape[0]):
            keypoint_variability_av = []
            for i in range(variability.shape[1]):
                start_index = frame_index-eachAverageTime
                if start_index < 0:
                    start_index = 0
                division = frame_index+1 - start_index
                keypoint_variability_av.append(
                    np.sum(variability[start_index:frame_index+1][..., i])/division)
            result.append(keypoint_variability_av)
        result = np.asarray(result)
        return result
