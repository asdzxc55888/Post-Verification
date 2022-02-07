import os
import numpy as np
import torch
from common.helpers import getFilesPath
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
import torchvision.transforms as trns
from PIL import Image
import torchvision.io as torch_io
import pandas as pd
from os import path
from tqdm import tqdm
import cv2

class generateVideoInfo:
    def __init__(self, media_path):
        self.media_path = media_path
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def __getVideoFrames(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def __predictVideoBboxes(self, video_frames):
        result = None
        batch_size = 100
        df= pd.DataFrame({})
# print(frames)
        for epoch in tqdm(range(round(len(video_frames)/batch_size + 1))):
            start_index = epoch * batch_size
            end_index = epoch * batch_size + batch_size
            result = self.model(video_frames[start_index:end_index])
    
            for i in range(len(result.pandas().xyxy)):
                singalDf = result.pandas().xyxy[i]
                singalDf['framesKey'] = i + start_index
                df = df.append(singalDf)

        isOverBaseConfidence = df["confidence"] > 0.7
        isPersonClass = df["class"] == 0
        df = df[(isOverBaseConfidence & isPersonClass )]

        return df.drop(['confidence', 'class', 'name'], axis=1).to_numpy()

    def generateVideoInfo(self, video_file_name):
        files_path = getFilesPath(path.join(self.media_path, video_file_name), 'mp4')
        
        videos_bboxes = []
        for file_path in files_path:
            video_frames = self.__getVideoFrames(file_path)
            videos_bboxes.append(self.__predictVideoBboxes(video_frames))


        
