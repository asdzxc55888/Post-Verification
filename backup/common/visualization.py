import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import os
import cv2

from tqdm import tqdm
from config import cfg
from common.humanKeypoints import humanKeypoints


class visualization:
    def __init__(self, keypoints: humanKeypoints) -> None:
        self.__video_path = os.path.join(
            cfg.video_path, keypoints.getVideoName())
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

    def save_keypoint_video(self, start_frame=0, end_frame=0):
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
        show_frames = []
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        bone_list = [[0, 1], [0, 2], [1, 3], [2, 4], [4, 6], [3, 5], [6, 8], [8, 10], [
            6, 12], [12, 14], [14, 16], [12, 11], [11, 13], [13, 15], [5, 11], [5, 7], [7, 9]]

        for i in tqdm(np.arange(start_frame, end_frame)):
            keypointsAx = []
            singal_result = self.__keypoints_df.iloc[i]
            if singal_result.size != 0:
                keypoints = singal_result['keypoints']
                count = 0
                for keypoint in keypoints:
                    keypointsAx.append(ax.scatter(
                        keypoint[0], keypoint[1], c="m"))
                    keypointsAx.append(ax.annotate(
                        count, (keypoint[0], keypoint[1])))
                    count = count + 1
                for bone_index in range(len(bone_list)):
                    bone1 = keypoints[bone_list[bone_index][0]]
                    bone2 = keypoints[bone_list[bone_index][1]]
                    newsegm, = ax.plot([bone1[0], bone2[0]], [
                                          bone1[1], bone2[1]], 'y')
                    keypointsAx.append(newsegm)
            keypointsAx.append(ax.imshow(
                frames[singal_result['framesKey']], animated=True))
            show_frames.append(keypointsAx)

        ffmpegpath = os.path.abspath(
            "/home/biolab/anaconda3/envs/Wei_env/bin/ffmpeg")
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
        writer = animation.FFMpegWriter(fps=24)

        ani = animation.ArtistAnimation(
            fig, show_frames, interval=1000/24, blit=True)
        file_name = self.__keypoints.getVideoName().split('.')[
            0] + '_result.mp4'
        save_path = os.path.join(cfg.output_video_path, file_name)
        ani.save(save_path, writer=writer)

    def draw_average_movement(self, start_frame: int = 0, end_frame: int = 0, title: str = 'Average movement', eachAverageTime: int = 72) -> None:
        '''
        draw average movement chart.

        Parameter
        -----
        start_frame: which frames do you want to start from.
        end_frame: which frames do you want to end in.
        title: the title of plot.
        '''
        plt.rcParams['font.sans-serif'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False

        average_movement = self.__keypoints.getAverageMovement(eachAverageTime)

        final_frame = self.__keypoints.getKeypoints().shape[0] - 1
        if end_frame == 0 or end_frame > final_frame:
            end_frame = final_frame
        if start_frame < 0 or start_frame > final_frame:
            start_frame = 0

        x = np.arange(start_frame, end_frame)

        fig, ax = plt.subplots()
        ax.set_title(title)
        range_average_movement = average_movement[start_frame:end_frame]
        ax.plot(x, range_average_movement[..., 9] +
                range_average_movement[..., 7], lw=2, label='left hand')
        ax.plot(x, range_average_movement[..., 10] +
                range_average_movement[..., 8], lw=2, label='right hand')
        ax.plot(x, range_average_movement[..., 15] +
                range_average_movement[..., 13], lw=2, label='left foot')
        ax.plot(x, range_average_movement[..., 16] +
                range_average_movement[..., 14], lw=2, label='right foot')
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
