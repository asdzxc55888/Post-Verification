import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib
import numpy as np
import pandas as pd
import os
import cv2

from tqdm import tqdm
from config import cfg
from common.humanKeypoints import humanKeypoints

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [
        8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [
                  0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
NUM_KPTS = 17


class visualization:
    def __init__(self, keypoints: humanKeypoints) -> None:
        self.__video_path = os.path.join(
            cfg.VIDEO_PATH, keypoints.getVideoName())
        self.__keypoints = keypoints
        self.__keypoints_df = keypoints.getKeypoints(isNormalize=False)
        pass

    def __getVideoFrames(self) -> list:
        cap = cv2.VideoCapture(self.__video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def save_keypoint_video(self, start_frame=0, end_frame=0, show_noise=False):
        '''
        Saving the video with keypoints information in the setting folder.

        Parameter
        ------
        start_frame: which frames do you want to start from.
        end_frame: which frames do you want to end in.
        '''

        if end_frame == 0 or end_frame > self.__keypoints_df.shape[0] - 1:
            end_frame = self.__keypoints_df.shape[0] - 1

        if start_frame < 0 or start_frame > self.__keypoints_df.shape[0] - 1:
            start_frame = 0

        frames = self.__getVideoFrames()
        noise_frame = self.__keypoints.getNoiseFrames(threshold=1)

        file_name = self.__keypoints.getVideoName().split('.')[
            0] + '_result.mp4'
        save_path = os.path.join(cfg.OUTPUT_VIDEO_PATH, file_name)

        vidcap = cv2.VideoCapture(self.__video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 24.0,
                              (int(vidcap.get(3)), int(vidcap.get(4))))
        for i in tqdm(np.arange(start_frame, end_frame)):

            if show_noise and noise_frame[i]:
                print("1")
                continue

            singal_result = self.__keypoints_df.iloc[i]
            keypoints = singal_result['keypoints']
            self.__draw_pose(np.asarray(keypoints),
                             frames[singal_result['framesKey']])
            frame = cv2.cvtColor(
                frames[singal_result['framesKey']], cv2.COLOR_BGR2RGB)
            out.write(frame)

    def draw_average_movement(self, start_frame: int = 0, end_frame: int = 0, title: str = 'Average movement', eachAverageTime: int = 24) -> None:
        '''
        draw average movement chart.

        Parameter
        -----
        start_frame: which frames do you want to start from.
        end_frame: which frames do you want to end in.
        title: the title of plot.
        '''

        average_movement = self.__keypoints.getAverageMovement(eachAverageTime)
        
        final_frame = self.__keypoints.getKeypoints().shape[0] - eachAverageTime
        if end_frame == 0 or end_frame > final_frame:
            end_frame = final_frame
        if start_frame < 0 or start_frame > final_frame:
            start_frame = 0

        x = np.arange(start_frame, end_frame)

        fig, ax = plt.subplots()
        ax.set_title(title)
        range_average_movement = average_movement[..., start_frame:end_frame]
        # ax.plot(x, range_average_movement[9] +
        #         range_average_movement[7], lw=2, label='left hand')
        # ax.plot(x, range_average_movement[10] +
        #         range_average_movement[8], lw=2, label='right hand')
        # ax.plot(x, range_average_movement[15] +
        #         range_average_movement[13], lw=2, label='left foot')
        # ax.plot(x, range_average_movement[16] +
        #         range_average_movement[14], lw=2, label='right foot')
        # ax.plot(x, range_average_movement[9], lw=2, label='keypoint 9')
        # ax.plot(x, range_average_movement[10], lw=2, label='keypoint 10')
        # ax.plot(x, range_average_movement[15], lw=2, label='keypoint 15')
        ax.plot(x, range_average_movement[16], lw=2, label='keypoint 16')
        plt.legend()
        plt.show()

    def draw_variability(self, start_frame: int = 0, end_frame: int = 0, title: str = 'Variability') -> None:
        '''
        draw variability chart.

        Parameter
        -----
        start_frame: which frames do you want to start from.
        end_frame: which frames do you want to end in.
        title: the title of plot.
        '''
        variability = self.__keypoints.getKeypointVariability()

        final_frame = self.__keypoints.getKeypoints().shape[0] - 1
        if end_frame == 0 or end_frame > final_frame:
            end_frame = final_frame
        if start_frame < 0 or start_frame > final_frame:
            start_frame = 0

        x = np.arange(start_frame, end_frame)

        fig, ax = plt.subplots()
        ax.set_title(title)
        range_variability = variability[start_frame:end_frame]
        ax.plot(x, range_variability[..., 9] +
                range_variability[..., 7], lw=2, label='left hand')
        ax.plot(x, range_variability[..., 10] +
                range_variability[..., 8], lw=2, label='right hand')
        ax.plot(x, range_variability[..., 15] +
                range_variability[..., 13], lw=2, label='left foot')
        ax.plot(x, range_variability[..., 16] +
                range_variability[..., 14], lw=2, label='right foot')
        plt.legend()
        plt.show()

    def draw_peak_feature(self, features_df, visualize_data, video_name):
        COLOR = ['red', 'orange', 'gold', 'magenta', 'green', 'turquoise','deepskyblue', 'indigo', 'black', 'pink']
        fig, ax = plt.subplots(3,4,figsize=(30, 15))

        for i, data in enumerate(visualize_data):
            joint_index = data['joint_index']
            segment = data['segment']
            peak = data['peak']
            moving_average = data['moving_average']

            ma_range = np.arange(moving_average.shape[0])
            clusters = features_df[(features_df['joint_index'] == joint_index) & (features_df['video_name'] == video_name)]

            row = int(i/4)
            col = i%4
            ax[row][col].set_title(joint_index)
            for j, s in enumerate(segment): 
                segment_range = np.arange(s[0], s[1])
                peakCluster = clusters["peakCluster"].iloc[j]
                ax[row][col].plot(segment_range, moving_average[segment_range], COLOR[peakCluster]); 
            # ax[row][col].plot(peak, moving_average[peak], "xr")
            ax[row][col].plot(segment, moving_average[segment], "dr"); 
            ax[row][col].plot(moving_average, 'black', alpha = 0.3); 
            # ax[row][col].legend(['peak'])
            ax[row][col].legend(['segment'])

    def __draw_pose(self, keypoints, img):
        """draw the keypoints and the skeletons.
        :params keypoints: the shape should be equal to [17,2]
        :params img:
        """
        assert keypoints.shape == (NUM_KPTS, 2)
        for i in range(len(SKELETON)):
            kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
            x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
            x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
            cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
            cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
            cv2.line(img, (int(x_a), int(y_a)),
                     (int(x_b), int(y_b)), CocoColors[i], 2)
