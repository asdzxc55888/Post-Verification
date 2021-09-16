import os
import numpy as np
import torch
from common.helpers import getFilesPath
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
import torchvision.transforms as trns
from PIL import Image
import cv2

class generateImageInfo:
    def __init__(self, media_path):
        self.media_path = media_path
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

    def __getVideoFrames(video_path):
        cap = cv2.VideoCapture(video_path)
        result = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            transform = trns.ToTensor()
            frame_image = transform(frame_image)
            result.append(frame_image)
        cap.release()
        return result

    def __predictVideoBboxes(self, video_frames):
        bboxes = []
        outputs = self.model(video_frames)
        for index in range(len(outputs)):
            outputs_np = {k: v.detach().numpy() for k, v in outputs[index].items()}
            for i, (bbox, label, score) in enumerate(zip(outputs_np["boxes"], outputs_np["labels"], outputs_np["scores"])):
                if label == 1 and score > 0.8:
                    bboxes.append(bbox)

        return bboxes

    def generateVideoInfo(self, result_path):
        files_path = getFilesPath(self.media_path, 'mp4')
        
        videos_bboxes = []
        for file_path in files_path:
            video_frames = self.__getVideoFrames(file_path)
            videos_bboxes.append(self.__predictVideoBboxes(video_frames))


        
