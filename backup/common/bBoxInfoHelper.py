import string
import pandas as pd
import torch
from math import floor
from tqdm import tqdm
import cv2
import os

from utils_local.osutils import mkdir_p, isdir
from config import cfg


class bBoxInfoHelper:
    '''
    Use pretrained yolov5 model to predict bbox position info of the human in the video.
    '''

    def __init__(self, video_name: string, batch_size: int = 16):
        self.__video_name = video_name
        self.video_path = os.path.join(cfg.video_path, video_name)
        self.batch_size = batch_size
        self.frames = self.__getVideoFrames()
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'yolov5m', pretrained=True)
        self.result = self.__predictVideoBboxes()

    def __getVideoFrames(self) -> list:
        print(self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def __saveInfoResult(self, result_df) -> None:
        if not isdir(cfg.keypoint_result_path):
            mkdir_p(cfg.keypoint_result_path)
        result_path = os.path.join(
            cfg.video_info_path, self.__video_name.split(".")[0] + '.csv')
        result_df.to_csv(result_path, index=False)

    def __predictVideoBboxes(self) -> pd.DataFrame:
        print("predict Video bboxes...")
        result = None
        batch_size = self.batch_size
        df = pd.DataFrame({})

        for epoch in tqdm(range(floor(len(self.frames)/batch_size + 1))):
            start_index = epoch * batch_size
            end_index = epoch * batch_size + batch_size
            result = self.model(self.frames[start_index:end_index])

            for i in range(len(result.pandas().xyxy)):
                singalDf = result.pandas().xyxy[i]
                singalDf['framesKey'] = i + start_index
                df = df.append(singalDf)

        isOverBaseConfidence = df["confidence"] > 0.7
        isPersonClass = df["class"] == 0

        filter_df = df[(isOverBaseConfidence & isPersonClass)]

        self.__saveInfoResult(filter_df)

        return filter_df

    def getVideoInfo(self) -> pd.DataFrame:
        '''
        Get video info with dataframe(pandas) type
        '''
        return self.result
